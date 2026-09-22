"""Sparse proximal code proposals driven by routed-output gradients.

This is a layer-local optimizer, not full-model published PV-Tuning. Each
expert/projection proposal is evaluated against actual training output loss.
"""
import math
import torch
from .activation import ARITHMETIC, activation_ste, swiglu_ste

METHOD = 'routed_output_gradient_sparse_v1'

@torch.no_grad()
def propose_codes(p, e, grad, max_fraction=.01, trust_ratio=.01, target_ratio=.1):
    """Search single-book alternatives near a gradient step; cap count and norm.

The candidate pool is the highest-gradient 4*budget groups. Search all 256
choices in either book, with the other book fixed. Score using the proximal
linearization g*delta + ||delta||^2/(2*eta). Original FP8 weights are never used.
    """
    w=p.weight(e).detach().reshape(-1,8);g=grad.detach().reshape(-1,8)
    if not torch.isfinite(g).all():raise ValueError('Nonfinite index gradient')
    gn=g.norm();wn=w.norm();budget=int(len(w)*max_fraction)
    if budget<1 or float(gn)==0 or float(wn)==0:return None
    eta=target_ratio*wn/gn
    pool=min(len(w),4*budget)
    selected=g.square().sum(1).topk(pool,sorted=False).indices
    cb,s,glob=p.quantized(e);cb=cb.detach();factor=(s.detach().repeat_interleave(16,1)*glob.detach()).flatten()
    old_a=p.codes_a[e].flatten();old_b=p.codes_b[e].flatten()
    scores=[];aa=[];bb=[];norms=[]
    for ix in selected.split(2048):
        a=old_a[ix].long();b=old_b[ix].long();f=factor[ix,None]
        current=w[ix];target=current-eta*g[ix]
        # Distances need only [chunk,256], not [chunk,256,8].
        best=[]
        for book,other in ((cb[:256],cb[256+b]),(cb[256:],cb[a])):
            residual=target-f*other
            distances=f.square()*book.square().sum(1)[None,:]-2*f*(residual@book.T)
            choice=distances.argmin(1)
            candidate=f*(book[choice]+other);delta=candidate-current
            cost=(g[ix]*delta).sum(1)+delta.square().sum(1)/(2*eta)
            best.append((choice,cost,delta.square().sum(1)))
        choose_a=best[0][1]<=best[1][1]
        scores.append(torch.minimum(best[0][1],best[1][1]))
        aa.append(torch.where(choose_a,best[0][0],a));bb.append(torch.where(choose_a,b,best[1][0]))
        norms.append(torch.where(choose_a,best[0][2],best[1][2]))
    score=torch.cat(scores);a=torch.cat(aa);b=torch.cat(bb);delta2=torch.cat(norms)
    valid=((score<0)&(delta2>0)).nonzero().flatten()
    if not len(valid):return None
    order=valid[score[valid].argsort()[:budget]]
    # Strict trust radius for the reconstructed projection, not latent values.
    order=order[delta2[order].cumsum(0)<=(trust_ratio*wn).square()]
    if not len(order):return None
    ix=selected[order]
    return {'positions':ix,'a':a[order].to(old_a.dtype),'b':b[order].to(old_b.dtype),
            'old_a':old_a[ix].clone(),'old_b':old_b[ix].clone(),
            'predicted_change':float(score[order].sum()),
            'relative_weight_change':float(delta2[order].sum().sqrt()/wn)}


def expert_output(z,w13,w2,arithmetic):
    if arithmetic==ARITHMETIC:
        gu=activation_ste(z)@w13.T
        return activation_ste(swiglu_ste(gu))@w2.T
    gu=z@w13.T;gate,up=gu.chunk(2,-1)
    return (torch.nn.functional.silu(gate)*up)@w2.T


@torch.no_grad()
def reassign_indices(p13,p2,x,rows,cold,args,target,prediction,routes):
    """Sequential expert/projection updates with exact routed residual checks.

Only the provided training rows enter gradients, candidate scoring or acceptance.
The routed residual includes every cold expert, so gates and cross-expert
error interactions are included. Validation/audit are inaccessible here.
    """
    ref=target[rows];residual=prediction(rows)-ref
    denominator=ref.square().sum().clamp_min(1e-20)
    before=float((residual.square().sum()/denominator).sqrt())
    accepted=proposed=changed=skipped=0;details=[]
    for e,eid in enumerate(cold):
        weights=routes[e][rows];use=weights!=0
        if int(use.sum())<args.reassign_min_rows:skipped+=1;continue
        z=x[rows[use]];gate=weights[use,None]
        for p,name in ((p13,'w13'),(p2,'w2')):
            w13=p13.weight(e).detach();w2=p2.weight(e).detach()
            with torch.enable_grad():
                weight=w13 if name=='w13' else w2;weight.requires_grad_(True)
                old_output=expert_output(z,w13,w2,args.arithmetic)
                # At current weights this equals the actual coupled output loss.
                routed=residual[use].detach()+gate*(old_output-old_output.detach())
                loss=routed.square().sum()/denominator
                grad,=torch.autograd.grad(loss,weight)
            old_output=old_output.detach();w13=w13.detach();w2=w2.detach()
            proposal=propose_codes(p,e,grad,args.reassign_max_fraction,
                                   args.reassign_trust_ratio,args.reassign_target_ratio)
            del grad,loss,routed,weight
            if proposal is None:continue
            proposed+=1;ix=proposal['positions']
            a=p.codes_a[e].view(-1);b=p.codes_b[e].view(-1)
            a[ix]=proposal['a'];b[ix]=proposal['b']
            try:
                new_output=expert_output(z,p13.weight(e).detach(),p2.weight(e).detach(),args.arithmetic)
                candidate=residual[use]+gate*(new_output-old_output)
                old_loss=float(residual[use].square().sum());new_loss=float(candidate.square().sum())
                keep=math.isfinite(new_loss) and new_loss<old_loss-max(1e-12,old_loss*1e-7)
                if keep:residual[use]=candidate;accepted+=1;changed+=len(ix)
                else:a[ix]=proposal['old_a'];b[ix]=proposal['old_b']
            except BaseException:
                a[ix]=proposal['old_a'];b[ix]=proposal['old_b'];raise
            details.append({'expert':eid,'projection':name,'accepted':keep,'changed_groups':len(ix),
                            'relative_weight_change':proposal['relative_weight_change'],
                            'training_before_local':old_loss,'training_candidate_local':new_loss})
    # Recompute rather than trust accumulated residual arithmetic. Rollback is
    # handled by the caller's full index snapshot if the net check regresses.
    after=float(((prediction(rows)-ref).square().sum()/denominator).sqrt())
    return {'method':METHOD,'training_before':before,'training_candidate':after,
            'accepted':math.isfinite(after) and after<before,
            'accepted_proposals':accepted,'proposals':proposed,'changed_groups':changed,
            'skipped_experts':skipped,'max_changed_fraction':args.reassign_max_fraction,
            'trust_ratio':args.reassign_trust_ratio,'details':details}
