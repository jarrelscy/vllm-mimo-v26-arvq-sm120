import torch
from arvqprep import pack as pk
from arvq88.encoder import EncodedLayerProj
class Projection(torch.nn.Module):
    def __init__(self,store,pre,proj,E,dev,codebook_scope='preserve',residual_scale_shift=0):
        super().__init__(); self.proj=proj
        data=store[proj]
        self.residual_scale_shift=int(residual_scale_shift)
        self.res_factor=2.0**(-self.residual_scale_shift)
        self.N,self.K=data['N'],data['K']
        self.scale_dtype=data.get('scale_dtype','fp8_e4m3')
        if self.scale_dtype not in ('fp8_e4m3','fp16'):raise ValueError('Unsupported block scale dtype')
        self.scale_torch_dtype=torch.float16 if self.scale_dtype=='fp16' else torch.float8_e4m3fn
        self.scale_max=65504 if self.scale_dtype=='fp16' else 448
        self.codebook_dtype=data.get('codebook_dtype','fp4_grid')
        if self.codebook_dtype not in ('fp4_grid','fp16'):raise ValueError('Unsupported codebook dtype')
        cb=torch.cat([data['c0'],data['c1']],dim=-2)
        if cb.ndim==3: cb=cb[:E]
        elif codebook_scope=='expert': cb=cb.unsqueeze(0).repeat(E,1,1)
        if cb.shape not in [(512,8),(E,512,8)]: raise ValueError('Invalid codebook shape')
        self.cb=torch.nn.Parameter(cb.to(dev))
        self.log_global=torch.nn.Parameter(torch.tensor([data['glob']],device=dev).log())
        self.register_buffer('codes_a',data['a'][:E].to(dev).clone())
        self.register_buffer('codes_b',data['b'][:E].to(dev).clone())
        self.log_scales=torch.nn.ParameterList([torch.nn.Parameter(data['s'][e].to(dev).float().clamp_min(1e-10).log()) for e in range(E)])
    def _project_cb(self,cb):
        if self.codebook_dtype=='fp16':return cb+(cb.half().float()-cb).detach()
        return cb+(pk.project_to_fp4(cb)-cb).detach()
    def quantized(self,e):
        cb=self.cb[e] if self.cb.ndim==3 else self.cb
        cb=self._project_cb(cb)
        raw=self.log_scales[e].exp().clamp(max=self.scale_max)
        scale=raw+(raw.to(self.scale_torch_dtype).float()-raw).detach()
        return cb,scale,self.log_global.exp()
    def weight(self,e):
        cb,s,g=self.quantized(e); a,b=self.codes_a[e],self.codes_b[e]
        w=(cb[a.long()]+self.res_factor*cb[256+b.long()])
        return (w*s.repeat_interleave(16,1)[:,:,None]*g).reshape(self.N,self.K)
    def encoded(self):
        cb=(self.cb.detach().half().float() if self.codebook_dtype=='fp16' else pk.project_to_fp4(self.cb.detach())).cpu()
        s=torch.stack([p.detach().exp().clamp(max=self.scale_max).to(self.scale_torch_dtype).float().cpu() for p in self.log_scales])
        return EncodedLayerProj(cb[..., :256, :],cb[..., 256:, :],float(self.log_global.detach().exp()),
            self.codes_a.cpu().clone(),self.codes_b.cpu().clone(),
            s,self.N,self.K,0.,0.,self.scale_dtype,self.codebook_dtype)
