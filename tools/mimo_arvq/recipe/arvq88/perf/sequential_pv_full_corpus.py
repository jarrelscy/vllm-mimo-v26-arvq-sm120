"""Experimental faster fork: larger microbatches and sparse proposal communication.

Each rank owns distinct experts. Routed outputs are summed; no dense-model
optimizer replica or full-model gradients are required. Blocks run sequentially.
"""
import argparse,copy,json,os,sys,time
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from arvq88.pv import Projection
from arvq88.gradient_indices import propose_codes,expert_output
from arvq88.activation import ARITHMETIC
from arvq88.encoder import EncodedLayerProj
from arvq88.inputs import write
from arvq88.perf.sparse_delta import sparse_broadcast_delta
from arvq88.perf.distributed_batch import broadcast_batch


class BlockProjection(Projection):
    def __init__(self,store,proj,slots,dev,scale_dtype="fp8_e4m3",residual_scale_shift=0):
        d=dict(store[proj]);d['scale_dtype']=scale_dtype;ix=torch.tensor(slots)
        for k in ('c0','c1','a','b','s'):d[k]=d[k][ix]
        super().__init__({proj:d},'',proj,len(slots),dev,residual_scale_shift=residual_scale_shift)
        self.log_global.requires_grad_(False)
        for p in self.log_scales:p.requires_grad_(True)
        self.register_buffer("row_delta",torch.zeros(len(slots),self.N,1,device=dev))
        # Residual rows are held in EFFECTIVE units (grid values * res_factor) so
        # Adam moves the residual reconstruction at full codebook_lr and
        # propose_codes' symmetric decode stays exact under residual_scale_shift.
        # encoded() converts back to grid units; a no-op when res_factor == 1.
        with torch.no_grad():self.cb[:,256:,:]*=self.res_factor
    def quantized(self,e):
        from arvqprep import pack as pk
        cb=self.cb[e]
        if self.res_factor==1.0:proj=pk.project_to_fp4(cb)
        else:proj=torch.cat([pk.project_to_fp4(cb[:256]),pk.project_to_fp4(cb[256:]/self.res_factor)*self.res_factor])
        cb=cb+(proj-cb).detach()
        raw=self.log_scales[e].exp().clamp(max=self.scale_max)
        s=raw+(raw.to(self.scale_torch_dtype).float()-raw).detach()
        return cb,s,self.log_global.exp().detach()
    def weight(self,e):
        cb,s,g=self.quantized(e)
        w=torch.nn.functional.embedding(self.codes_a[e].long(),cb[:256])+torch.nn.functional.embedding(self.codes_b[e].long(),cb[256:])
        return (w*s.repeat_interleave(16,1)[:,:,None]*g).reshape(self.N,self.K)
    def encoded(self):
        values=[self.quantized(e) for e in range(len(self.log_scales))]
        cb=torch.stack([v[0].detach().cpu() for v in values]);s=torch.stack([v[1].detach().cpu() for v in values])
        if self.res_factor!=1.0:cb=torch.cat([cb[:,:256],cb[:,256:]/self.res_factor],dim=1)
        return EncodedLayerProj(cb[:,:256],cb[:,256:],float(self.log_global.exp()),self.codes_a.cpu().clone(),self.codes_b.cpu().clone(),s,self.N,self.K,0.,0.,self.scale_dtype)


def distributed_sum(local,gradient=False):
    summed=local.detach().clone();dist.all_reduce(summed)
    return local+(summed-local.detach()) if gradient else summed


def scalar_sum(value,device):
    t=torch.tensor(value,device=device,dtype=torch.float64);dist.all_reduce(t);return float(t)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',required=True);ap.add_argument('--layer',type=int,required=True)
    ap.add_argument('--target',choices=['reference','same_input'],default='reference');ap.add_argument('--steps',type=int,default=200)
    ap.add_argument('--reassign-every',type=int,default=20)
    ap.add_argument('--microbatch',type=int,default=65536)
    ap.add_argument('--batch-tokens',type=int,default=65536)
    ap.add_argument('--codebook-lr',type=float,default=.003)
    ap.add_argument('--scale-lr',type=float,default=.002)
    ap.add_argument('--benchmark-batches',action='store_true')
    ap.add_argument('--resume',help='Complete training checkpoint directory')
    ap.add_argument('--refine-from',help='Start another corpus pass from a retained best-step checkpoint, preserving Adam moments')
    ap.add_argument('--sequence-seed',type=int,default=7103)
    ap.add_argument('--export-only',action='store_true',help='Export retained best from resume without further updates')
    ap.add_argument('--mixed-validation',help='Additional held-out capture; first matched_validation_rows rows select checkpoints')
    ap.add_argument('--match-200-report',help='Require replay to match prior step-200 validation')
    a=ap.parse_args()
    if a.microbatch<1 or a.microbatch>a.batch_tokens:raise ValueError('Invalid microbatch')
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);dev=f'cuda:{rank}'
    torch.cuda.set_device(rank);torch.set_num_threads(2);dist.init_process_group('nccl');torch.manual_seed(91426)
    w=Path(a.work);cfg=json.loads((w/'config.json').read_text());base=Path(cfg['baseline'])
    reassign_kw=dict(max_fraction=cfg.get('reassign_max_fraction',.001),trust_ratio=cfg.get('reassign_trust_ratio',.01),target_ratio=cfg.get('reassign_target_ratio',.03))
    policy_path=w/'adaptive_schedule.json'
    if policy_path.exists():
        policy=json.loads(policy_path.read_text())
        if a.layer>=policy['from_layer']:cfg['adaptive_schedule']=policy
    out=w/f'layer{a.layer}_{a.target}';out.mkdir(exist_ok=True)
    source=base/'initial'/f'layer_{a.layer:05d}';manifest=json.loads((source/'arvq-manifest.json').read_text())
    all_cold=manifest['cold_expert_ids'];slots=list(range(rank,len(all_cold),world));cold=[all_cold[e] for e in slots]
    store={}
    for name in ('w13','w2'):store.update(torch.load(source/f'{name}.pt',weights_only=True,mmap=True))
    _rss=int(cfg.get('residual_scale_shift',0))
    p13=BlockProjection(store,'w13',slots,dev,cfg.get('block_scale_dtype','fp8_e4m3'),_rss);p2=BlockProjection(store,'w2',slots,dev,cfg.get('block_scale_dtype','fp8_e4m3'),_rss);del store
    parts=[torch.load(w/f'capture{a.layer}'/f'rank{i}.pt',weights_only=True,mmap=True) for i in range(world)]
    lengths=[len(p['x']) for p in parts]
    def merged(k):return torch.cat([p[k].to(dev) for p in parts])
    x=merged('x');ids=merged('topk_ids');gates=merged('topk_weights');labels=merged('pv_split')
    frozen=merged('frozen_output');reference=merged('reference_output');same=merged('same_input_output')
    eval_boundary=torch.cat([p['row_weight'].to(dev).float() for p in parts]) if 'row_weight' in parts[0] else None
    sequence_ids=torch.cat([p['sequence_ids'] for p in parts]);del parts
    if a.mixed_validation:
        mixed=torch.load(a.mixed_validation,weights_only=True,mmap=True);n=cfg['matched_validation_rows']
        x=torch.cat([x,mixed['x'][:n].to(dev).float()]);ids=torch.cat([ids,mixed['topk_ids'][:n].to(dev)]);gates=torch.cat([gates,mixed['topk_weights'][:n].to(dev)])
        frozen=torch.cat([frozen,mixed['frozen_output'][:n].to(dev)]);reference=torch.cat([reference,mixed['reference_output'][:n].to(dev)]);same=torch.cat([same,mixed['reference_output'][:n].to(dev)])
        labels=torch.cat([labels,torch.full((n,),3,dtype=labels.dtype,device=dev)])
        if eval_boundary is not None:eval_boundary=torch.cat([eval_boundary,torch.ones(n,device=dev)])
        del mixed
    target=reference if a.target=='reference' else same
    required=target-frozen
    from arvq88.perf.full_corpus import FullCorpus
    corpus=FullCorpus(w,a.batch_tokens,seed=a.sequence_seed,layer=a.layer if (w/f'training_capture{a.layer}').exists() else None);a.steps=corpus.steps
    train=range(corpus.rows);reasoning_val,audit=[(labels==i).nonzero().flatten() for i in (1,2)]
    val=(labels==3).nonzero().flatten() if a.mixed_validation else reasoning_val
    routes=[(gates*(ids==eid)).sum(1) for eid in cold]
    coverage=corpus.coverage[cold].to(dev)
    eligible=coverage>=256
    # Sparse experts retain initial books and scale corrections. Expert-specific
    # parameters make these masks exact; no optimizer state exists before masking.
    for p in (p13,p2):
        p.cb.register_hook(lambda g:g*eligible[:,None,None])
        for e,param in enumerate(p.log_scales):param.register_hook(lambda g,e=e:g*eligible[e])
    anchors=[p.cb.detach().clone() for p in (p13,p2)]
    scale_params=list(p13.log_scales)+list(p2.log_scales)
    scale_anchors=[p.detach().clone() for p in scale_params]
    trainable=[p13.cb,p2.cb]+scale_params
    optimizer=torch.optim.Adam([{'params':[p13.cb,p2.cb],'lr':a.codebook_lr},
                                {'params':scale_params,'lr':a.scale_lr}])
    denominator=torch.tensor(corpus.denominator,device=dev)
    evaluation=(x,ids,gates,frozen,reference,target,required,routes)
    train_w=None
    def activate(data=None):
        nonlocal x,ids,gates,frozen,reference,target,required,routes,train_w
        if data is None:
            x,ids,gates,frozen,reference,target,required,routes=evaluation;train_w=None
        else:
            x=ids=gates=frozen=reference=target=required=routes=None
            x=data['x'].to(dev).float();ids=data['topk_ids'].to(dev);gates=data['topk_weights'].to(dev).float();required=data['required'].to(dev).float()
            reference=target=required;frozen=None
            routes=[(gates*(ids==eid)).sum(1) for eid in cold]
            train_w=data['row_weight'].to(dev).float() if 'row_weight' in data else None
    probe_data=corpus.get_sequences(corpus.order[:4])
    def training_probe():
        activate(probe_data)
        value=evaluate(torch.arange(len(x),device=dev))['target_rel']
        activate()
        return value
    initial_state=[copy.deepcopy(p.state_dict()) for p in (p13,p2)]
    initial_quantized=[[(cb.detach().clone(),sc.detach().clone()) for cb,sc,g in [p.quantized(e) for e in range(len(p.log_scales))]] for p in (p13,p2)]
    @torch.no_grad()
    def movement():
        counts=torch.zeros(4,device=dev,dtype=torch.int64)
        for p,baseline in zip((p13,p2),initial_quantized):
            for e,(old_cb,old_sc) in enumerate(baseline):
                cb,sc,_=p.quantized(e)
                counts+=torch.stack([(cb!=old_cb).sum(),torch.tensor(cb.numel(),device=dev),(sc!=old_sc).sum(),torch.tensor(sc.numel(),device=dev)])
        dist.all_reduce(counts)
        v=counts.tolist()
        return {'codebook_changed':v[0],'codebook_values':v[1],'scales_changed':v[2],'scale_values':v[3]}

    eval_weights=None
    @contextmanager
    def cached_weights():
        nonlocal eval_weights
        if torch.is_grad_enabled():raise RuntimeError('Weight cache is evaluation-only')
        if eval_weights is not None:
            yield
            return
        eval_weights=[(p13.weight(e),p2.weight(e)) for e in range(len(cold))]
        try:yield
        finally:eval_weights=None
    def local_prediction(rows):
        y=torch.zeros(len(rows),6144,device=dev)
        for e,r in enumerate(routes):
            use=r[rows]!=0
            if not use.any():continue
            z=x[rows[use]]
            weights=eval_weights[e] if eval_weights is not None else (p13.weight(e),p2.weight(e))
            yy=expert_output(z,*weights,ARITHMETIC)
            y=y.index_add(0,use.nonzero().flatten(),yy*r[rows[use],None])
        return y
    def prediction(rows,gradient=False):return distributed_sum(local_prediction(rows),gradient)
    @torch.no_grad()
    def evaluate(rows,details=False,split=False):
        windows=[]
        sums=torch.zeros(4,device=dev,dtype=torch.float64)
        bsums=torch.zeros(4,device=dev,dtype=torch.float64)  # boundary_sse,boundary_energy,bulk_sse,bulk_energy
        do_split=split and eval_boundary is not None
        with cached_weights():
            for r in rows.split(1024):
                y=prediction(r)
                if frozen is not None:y=y+frozen[r]
                if details:
                    windows.append({'first_row':int(r[0]),'rows':len(r),
                        'reference_sse':float((y-reference[r]).square().sum()),
                        'reference_energy':float(reference[r].square().sum())})
                se=(y-target[r]).square()
                sums+=torch.stack([se.sum(),target[r].square().sum(),
                                    (y-reference[r]).square().sum(),reference[r].square().sum()]).double()
                if do_split:
                    bm=eval_boundary[r]>1
                    bsums+=torch.stack([se[bm].sum(),target[r][bm].square().sum(),se[~bm].sum(),target[r][~bm].square().sum()]).double()
        result={'target_rel':float((sums[0]/sums[1]).sqrt()),'reference_rel':float((sums[2]/sums[3]).sqrt())}
        if do_split:
            result['boundary_rel']=float((bsums[0]/bsums[1]).sqrt()) if bsums[1]>0 else None
            result['bulk_rel']=float((bsums[2]/bsums[3]).sqrt()) if bsums[3]>0 else None
            result['boundary_rows']=int((eval_boundary[rows]>1).sum())
        if details:result['windows']=windows
        return result
    @torch.no_grad()
    def discrete(rows,check_rows):
        # All ranks see the same coupled residual; each proposes for its own expert
        # in parallel, then proposal deltas are accepted serially across owners.
        both=torch.cat([rows,check_rows]);n=len(rows)
        residual=prediction(both)-required[both]
        before=float(residual.square().sum());accepted=proposed=groups=0
        counts=torch.tensor(len(cold),device=dev);dist.all_reduce(counts,op=dist.ReduceOp.MAX)
        for e in range(int(counts)):
            valid=e<len(cold) and bool(eligible[e])
            if valid:
                r=routes[e][both];use=r!=0;where=use.nonzero().flatten();proposal_mask=where<n
                valid=int(proposal_mask.sum())>=32
            for p,name in ((p13,'w13'),(p2,'w2')):
                proposal=None;old_output=None
                if valid:
                    z=x[both[use]];weight=r[use,None]
                    w13=p13.weight(e).detach();w2=p2.weight(e).detach()
                    with torch.enable_grad():
                        ww=w13 if name=='w13' else w2;ww.requires_grad_(True)
                        output=expert_output(z,w13,w2,ARITHMETIC)
                        residual_graph=residual[use].detach()+weight*(output-output.detach())
                        loss=residual_graph[proposal_mask].square().sum()/(denominator*n)
                        grad,=torch.autograd.grad(loss,ww)
                    old_output=output.detach()
                    proposal=propose_codes(p,e,grad,**reassign_kw)
                    del grad,ww,loss,residual_graph,output,w13,w2
                # Prefix backtracking is training-only. All ranks execute the same
                # collective schedule even if an expert has no valid proposal.
                for owner in range(world):
                    has=torch.tensor(int(proposal is not None) if rank==owner else 0,device=dev)
                    dist.broadcast(has,owner)
                    if not int(has):continue
                    proposed+=1
                    kept=False
                    for divisor in (1,4,16,10**9):
                        delta=torch.zeros_like(residual);changed=torch.zeros((),device=dev,dtype=torch.long)
                        if rank==owner:
                            k=max(1,len(proposal['positions'])//divisor);ix=proposal['positions'][:k]
                            aa=p.codes_a[e].view(-1);bb=p.codes_b[e].view(-1)
                            aa[ix]=proposal['a'][:k];bb[ix]=proposal['b'][:k]
                            yy=expert_output(z,p13.weight(e).detach(),p2.weight(e).detach(),ARITHMETIC)
                            delta[use]=weight*(yy-old_output);changed.fill_(k)
                        sparse_broadcast_delta(delta,where if rank==owner else None,owner);dist.broadcast(changed,owner)
                        candidate=residual+delta
                        tr0=float(residual[:n].square().sum());tr1=float(candidate[:n].square().sum())
                        ck0=float(residual[n:].square().sum());ck1=float(candidate[n:].square().sum())
                        kept=bool(torch.isfinite(candidate).all()) and tr1<tr0*(1-1e-7) and ck1<=ck0*(1+1e-7)
                        if kept:
                            residual=candidate;accepted+=1;groups+=int(changed);break
                        if rank==owner:
                            aa[ix]=proposal['old_a'][:k];bb[ix]=proposal['old_b'][:k]
                    if rank==owner and kept:proposal=None
        return {'method':'expert_parallel_output_gradient_prefix_backtracking_v1','proposals':proposed,
                'accepted':accepted,'changed_groups':groups,'training_before':before,
                'training_after':float(residual.square().sum()),'acceptance':'proposal and separate training check batches'}
    if a.benchmark_batches:
        from arvq88.perf.batch_benchmark import benchmark
        benchmark(p13,p2,optimizer,train,prediction,required,denominator,anchors,scale_params,scale_anchors,dev,out)
        dist.barrier();dist.destroy_process_group();return
    refinement=None
    if a.refine_from:
        if a.resume:raise ValueError('Use either resume or refine-from')
        checkpoint=Path(a.refine_from)
        if not (checkpoint/'complete.json').exists():raise ValueError('Incomplete refinement checkpoint')
        saved=torch.load(checkpoint/f'rank{rank}.pt',map_location=dev,weights_only=True)
        old=saved['signature']
        for key,value in [('layer',a.layer),('target',a.target),('world',world),('cold_ids',all_cold),('batch_tokens',a.batch_tokens),('microbatch',a.microbatch)]:
            if old[key]!=value:raise ValueError(f'Refinement mismatch: {key}')
        for key in ('baseline','corpus','source_cache','sequence_length','train_token_file'):
            if old['config'][key]!=cfg[key]:raise ValueError(f'Refinement data mismatch: {key}')
        if saved['step']!=saved['best_step']:raise ValueError('Refine from the checkpoint at the retained best step')
        for p,sd in zip((p13,p2),saved['current']):p.load_state_dict(sd)
        optimizer.load_state_dict(saved['optimizer'])
        optimizer.param_groups[0]['lr']=a.codebook_lr;optimizer.param_groups[1]['lr']=a.scale_lr
        initial_state=[copy.deepcopy(p.state_dict()) for p in (p13,p2)]
        refinement={'checkpoint':str(checkpoint),'parent_step':saved['step'],'parent_validation':saved['best'],'adam_moments_preserved':True,'sequence_seed':a.sequence_seed}
        del saved
    if rank==0:write(out/'status.json',{'stage':'evaluating_initial','experts':len(all_cold),'world':world})
    initial_audit=evaluate(audit,True,split=True) if cfg.get('record_initial_audit') else None
    initial_reasoning=evaluate(reasoning_val,True) if a.mixed_validation else None
    initial=evaluate(val,True,split=True);best=initial['target_rel'];state=initial_state;history=[];reassign=[];best_step=0;overfit=0
    if refinement and abs(initial['target_rel']/refinement['parent_validation']-1)>1e-3:raise ValueError('Refinement checkpoint replay failed')
    if rank==0 and refinement:write(out/'refinement.json',{**refinement,'replayed_validation':initial['target_rel']})
    generator=torch.Generator(device=dev).manual_seed(a.sequence_seed)
    train_at_best=training_probe();t0=time.time();start_step=0
    signature={'layer':a.layer,'target':a.target,'world':world,'codebook_lr':a.codebook_lr,'scale_lr':a.scale_lr,
               'microbatch':a.microbatch,'reassign_every':a.reassign_every,'config':cfg,'cold_ids':all_cold,'sequence_order_sha256':corpus.order_sha256,'batch_tokens':a.batch_tokens}
    if a.resume:
        resume=Path(a.resume)
        if not (resume/'complete.json').exists():raise ValueError('Incomplete training checkpoint')
        saved=torch.load(resume/f'rank{rank}.pt',map_location=dev,weights_only=True)
        if saved['signature']!=signature:raise ValueError('Resume configuration mismatch')
        for p,sd in zip((p13,p2),saved['current']):p.load_state_dict(sd)
        optimizer.load_state_dict(saved['optimizer']);generator.set_state(saved['generator'].cpu())
        state=saved['best_state'];best=saved['best'];best_step=saved['best_step'];overfit=saved['overfit']
        train_at_best=saved['train_at_best'];history=saved['history'];reassign=saved['reassign'];start_step=saved['step']
        torch.set_rng_state(saved['cpu_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),dev)
        del saved
    def save_training(step):
        folder=out/'training_checkpoints'/f'step_{step:05d}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'rank{rank}.pt';tmp=path.with_suffix('.tmp')
        torch.save({'signature':signature,'step':step,'current':[p.state_dict() for p in (p13,p2)],
                    'optimizer':optimizer.state_dict(),'generator':generator.get_state(),
                    'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(dev),
                    'best_state':state,'best':best,'best_step':best_step,'overfit':overfit,'train_at_best':train_at_best,
                    'history':history,'reassign':reassign},tmp);tmp.replace(path)
        dist.barrier()
        if rank==0:write(folder/'complete.json',{'step':step,'world_size':world,'complete':True})
    adaptive=None
    if cfg.get('adaptive_schedule'):
        from arvq88.perf.adaptive_schedule import AdaptiveSchedule
        policy=cfg['adaptive_schedule'];adaptive=AdaptiveSchedule(initial['target_rel'],cfg.get('lr_decay_after',45),policy['relative_worsening'],policy['checks'],policy['patience_updates'])
        for h in history:adaptive.observe(h['step'],h['validation']['target_rel'])
    step=start_step
    timings=[];batch_ready=time.perf_counter()
    batches=corpus.batches(start_step+1) if rank==0 else ((step,None) for step in range(start_step+1,a.steps+1))
    if a.export_only:
        if not a.resume:raise ValueError('export-only requires resume')
        batches=()
    for step,batch_data in batches:
        factor=cfg.get('lr_decay_factor',.25) if step>(adaptive.drop_after if adaptive else cfg.get('lr_decay_after',10**9)) else 1.
        optimizer.param_groups[0]['lr']=a.codebook_lr*factor
        optimizer.param_groups[1]['lr']=a.scale_lr*factor
        timing={'step':step,'data_wait':time.perf_counter()-batch_ready};stage_start=time.perf_counter()
        if rank==0 and (step==1 or step%10==0):write(out/'status.json',{'stage':'tuning','step':step,'steps':a.steps,'seconds':time.time()-t0})
        activate()  # Release the prior large GPU batch before allocating the next.
        batch_data=broadcast_batch(batch_data,dev)
        activate(batch_data);del batch_data
        torch.cuda.synchronize();timing['transfer']=time.perf_counter()-stage_start;stage_start=time.perf_counter()
        rows=torch.arange(len(x),device=dev)
        optimizer.zero_grad(set_to_none=True)
        # Microbatching preserves the single coupled output objective. Boundary rows
        # (next token in {</think>,<|endoftext|>}) are up-weighted; the normalizer is
        # the summed weight so the loss scale matches the unweighted objective at w=1.
        wsum=float(train_w.sum()) if train_w is not None else float(len(rows))
        total=0.
        for micro in rows.split(a.microbatch):
            pred=prediction(micro,True)
            diff=(pred-required[micro]).square()
            if train_w is not None:diff=diff*train_w[micro,None]
            loss=diff.sum()/(denominator*wsum)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();total+=float(loss.detach())
            del pred,loss,diff
        regularizer=(sum((p.cb-anchor).square().sum()/anchor.square().sum().clamp_min(1.)
                        for p,anchor in zip((p13,p2),anchors))
                     +sum((p-anchor).square().mean() for p,anchor in zip(scale_params,scale_anchors))/len(scale_params))*.01
        regularizer.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step()
        with torch.no_grad():
            for p in (p13,p2):p.cb.clamp_(-6,6)
            for p,anchor in zip(scale_params,scale_anchors):p.copy_(torch.maximum(torch.minimum(p,anchor+.35),anchor-.35))
        torch.cuda.synchronize();timing['continuous']=time.perf_counter()-stage_start;stage_start=time.perf_counter()
        if a.reassign_every and step%a.reassign_every==0:
            shuffled=rows[torch.randperm(len(rows),generator=generator,device=dev)[:8192]]
            result=discrete(shuffled[:4096],shuffled[4096:]);result['step']=step;reassign.append(result)
            if rank==0:print('REASSIGN',json.dumps(result),flush=True)
        torch.cuda.synchronize();timing['reassignment']=time.perf_counter()-stage_start;stage_start=time.perf_counter()
        if step%5==0 or step==a.steps:
            activate()
            metric=evaluate(val,split=True);reasoning_metric=evaluate(reasoning_val) if a.mixed_validation else None;tr=training_probe();changed=movement()
            if metric['target_rel']<best:
                best=metric['target_rel'];best_step=step;state=[copy.deepcopy(p.state_dict()) for p in (p13,p2)];train_at_best=tr
            adaptive_stop=adaptive.observe(step,metric['target_rel']) if adaptive else False
            diverging=metric['target_rel']>best*1.005 and tr<train_at_best*.995
            overfit=overfit+1 if diverging else 0
            row={'step':step,'validation':metric,'reasoning_validation':reasoning_metric,'training_probe_rel':tr,'batch_loss':total,'best_step':best_step,
                 'overfit_checks':overfit,'seconds':time.time()-t0,'adaptive_schedule':dict(vars(adaptive)) if adaptive else None,'adaptive_stop':adaptive_stop};history.append(row)
            if rank==0:
                row['quantized_movement']=changed
                print('PROGRESS',json.dumps(row),flush=True);write(out/'status.json',{'stage':'tuning',**row})
            if step==200 and a.match_200_report:
                expected=json.loads(Path(a.match_200_report).read_text())['history'][-1]['validation']['reference_rel']
                if abs(metric['reference_rel']/expected-1)>1e-3:raise ValueError('Step-200 replay exceeded 0.1 percent validation tolerance')
                if rank==0:write(out/'replay_check.json',{'passed':True,'expected':expected,'actual':metric['reference_rel'],'relative_difference':metric['reference_rel']/expected-1,'relative_tolerance':1e-3,'scope':'Numerical-quality replay check, not bitwise equivalence'})
            if step%5==0 or step==a.steps:save_training(step)
            if adaptive_stop or (step>=150 and overfit>=3):break
        torch.cuda.synchronize();timing['validation_checkpoint']=time.perf_counter()-stage_start;timings.append(timing)
        if rank==0:write(out/'timings.json',timings)
        batch_ready=time.perf_counter()
    activate()
    for p,s in zip((p13,p2),state):p.load_state_dict(s)
    final=evaluate(val,True,split=True);final_reasoning=evaluate(reasoning_val,True) if a.mixed_validation else None;audit_result=evaluate(audit,True,split=True);retained_movement=movement()
    encoded={key:vars(p.encoded()) for key,p in (('w13',p13),('w2',p2))}
    torch.save({'slots':slots,'encoded':encoded},out/f'experts_rank{rank}.pt')
    dist.barrier()
    # Exported state must reproduce the retained forward before trajectory use.
    reloaded=torch.load(out/f'experts_rank{rank}.pt',weights_only=True,mmap=True)['encoded']
    q13=Projection(reloaded,'','w13',len(slots),dev,residual_scale_shift=_rss);q2=Projection(reloaded,'','w2',len(slots),dev,residual_scale_shift=_rss)
    p13,p2=q13,q2
    serialized=evaluate(val)
    if abs(serialized['target_rel']-final['target_rel'])>2e-6:raise ValueError('Exported forward mismatch')
    outputs=[]
    with torch.no_grad(),cached_weights():
        for rows in torch.arange(sum(lengths),device=dev).split(1024):
            y=frozen[rows]+prediction(rows)
            if rank==0:outputs.append(y.to(torch.bfloat16).cpu())
    if rank==0:
        torch.save({'output':torch.cat(outputs),'rank_lengths':lengths,'sequence_ids':sequence_ids},out/'student_outputs.pt')
        report={'layer':a.layer,'target':a.target,'recipe':'full_corpus_sequence_pass_v1','block_scale_dtype':p13.scale_dtype,'residual_scale_shift':_rss,'batch_tokens':a.batch_tokens,'sequence_order_sha256':corpus.order_sha256,'training_sequence_count':len(corpus.order),'training_passes':min(step*a.batch_tokens,corpus.rows)/corpus.rows,'training_tokens_seen':min(step*a.batch_tokens,corpus.rows),'adaptive_schedule':dict(vars(adaptive)) if adaptive else None,'refinement':refinement,'lr_decay_after':cfg.get('lr_decay_after'),'lr_decay_factor':cfg.get('lr_decay_factor'),'export_only':a.export_only,'training_output_export':'fixed validation/audit outputs only; full training trajectory export not performed','codebook_lr':a.codebook_lr,'scale_lr':a.scale_lr,'quantized_movement':retained_movement,'initial_validation':initial,'final_validation':final,'initial_reasoning_validation':initial_reasoning,'final_reasoning_validation':final_reasoning,'validation_distribution':'document-disjoint mixture',
                'serialized_validation':serialized,'development_audit':audit_result,'initial_development_audit':initial_audit,'best_step':best_step,'steps_run':step,
                'history':history,'index_reassignments':reassign,'coverage_threshold':256,'checkpoint_selection':'held-out same-input MoE output','world_size':world,'microbatch':a.microbatch,'proposal_communication':'sparse routed rows','parallelism':'expert-sharded coupled output',
                'arithmetic':ARITHMETIC,'scope':'text-only, cached prompt partitions; historical audit is development only',
                'train_rows':len(train),'validation_rows':len(val),'audit_rows':len(audit),'complete':True}
        write(out/'report.json',report);write(out/'status.json',{'stage':'complete','best_step':best_step,'validation':final})
        print('COMPLETE',a.layer,a.target,initial['reference_rel'],final['reference_rel'],best_step,flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
