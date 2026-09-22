"""Serving activation-plane reference, pinned to b1380cf74d170b69a8708f1b1287cc09a6eb7df4.

Matches hybrid.cu pack_planes and the default FP16 SwiGLU path. This emulates
quantization boundaries, not SM120 MMA instruction accumulation bit-for-bit.
"""
import torch
REFERENCE_REVISION='b1380cf74d170b69a8708f1b1287cc09a6eb7df4'
ARITHMETIC='fp4_planes4_fp16_boundaries_v1'

def fp16_ste(x):
 rounded=x.to(torch.float16).float()
 return x+(rounded-x).detach() if x.requires_grad else rounded

def planes(x):
 """Return FP4 codes, FP8 scale bytes, and decoded *unweighted* planes."""
 if x.shape[-1]%16:raise ValueError('Activation input dimension must be divisible by 16')
 v=x.to(torch.float16).float().reshape(*x.shape[:-1],-1,16)
 if not torch.isfinite(v).all():raise ValueError('Nonfinite FP16 serving activation')
 cuts=torch.tensor([.25,.75,1.25,1.75,2.5,3.5,5.],device=x.device)
 table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
 codes=[];scales=[];decoded=[]
 for _ in range(4):
  peak=v.abs().amax(-1,keepdim=True)
  exponent=torch.ceil(torch.log2((peak/6).clamp_min(2**-20))).clamp(-6,8)
  scale=torch.exp2(exponent)
  # CUDA compares strictly > each midpoint: ties go toward zero, not ties-to-even.
  q=(v.abs().div(scale).unsqueeze(-1)>cuts).sum(-1)
  dec=table[q]*scale*torch.where(v<0,-1.,1.)
  codes.append((q|((v<0).long()<<3)).to(torch.uint8).reshape(x.shape))
  scales.append(((exponent.squeeze(-1).long()+7)<<3).to(torch.uint8))
  decoded.append(dec.reshape(x.shape))
  v=(v-dec)*16
 return torch.stack(codes),torch.stack(scales),torch.stack(decoded)

def activation_ste(x):
 with torch.no_grad():
  _,_,p=planes(x)
  result=p[0]+p[1]/16+p[2]/256+p[3]/4096
 return x+(result-x).detach() if x.requires_grad else result

def swiglu_ste(gu):
 # Serving: FP32 projection -> FP16; SiLU FP16 result; FP16 product.
 gu=fp16_ste(gu);gate,up=gu.chunk(2,-1)
 return fp16_ste(fp16_ste(torch.nn.functional.silu(gate))*up)
