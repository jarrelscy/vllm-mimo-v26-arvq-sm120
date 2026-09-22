"""ARVQ 8+8 test format: natural indices -> MMA fragments, 64 uint32 words/tile."""
import torch,json,hashlib
from pathlib import Path
LEVELS=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
def maps(device):
 p=torch.arange(128,device=device);j=p//32;lane=p%32
 return lane//4+8*(j%2),(j//2)*4+lane%4
def pack(a,b,N,K):
 r,k=maps(a.device);rows=torch.arange(N//16,device=a.device)[:,None,None]*16+r[None,None,:];cols=torch.arange(K//64,device=a.device)[None,:,None]*8+k[None,None,:]
 v=(a.long()|(b.long()<<8))[rows,cols]
 return (v[...,::2]|(v[...,1::2]<<16)).to(torch.uint32)
def unpack(w,N,K):
 w=w.long();v=torch.stack([w&65535,w>>16],-1).reshape(N//16,K//64,128)
 r,k=maps(w.device);rows=(torch.arange(N//16,device=w.device)[:,None,None]*16+r[None,None,:]).expand_as(v);cols=(torch.arange(K//64,device=w.device)[None,:,None]*8+k[None,None,:]).expand_as(v)
 a=torch.empty(N,K//8,dtype=torch.uint8,device=w.device);b=torch.empty_like(a)
 a[rows,cols]=(v&255).byte();b[rows,cols]=(v>>8).byte();return a,b
def pack_codebooks(c0,c1):
 cb=torch.cat([c0,c1],dim=-2)
 if cb.shape[-2:]!=(512,8) or cb.ndim not in (2,3):raise ValueError('Expected shared or per-expert 8+8 books')
 levels=torch.tensor(LEVELS,device=cb.device);n=(cb[...,None]-levels).abs().argmin(-1)
 if not torch.equal(levels[n],cb):raise ValueError('Codebooks must be exactly on the FP4 grid')
 return (n.long()<<(torch.arange(8,device=cb.device)*4)).sum(-1).to(torch.uint32)
def pack_codebooks_mb(cb,factors):
 """mcbook16: cb [E,256+B*256,8] in EFFECTIVE units. Rows are stored as plain-grid
 nibbles; decode multiplies residual book m by factors[m] (exact powers of two)."""
 B=len(factors)
 if cb.ndim!=3 or cb.shape[1]!=256+B*256 or cb.shape[2]!=8:raise ValueError('Invalid mcbook codebook shape')
 f=torch.cat([torch.ones(256),torch.tensor(factors).float().repeat_interleave(256)])[None,:,None]
 grid=cb/f
 levels=torch.tensor(LEVELS,device=cb.device);n=(grid[...,None]-levels).abs().argmin(-1)
 if not torch.equal(levels[n]*f,cb):raise ValueError('Codebooks must be exactly on their per-book grids')
 return (n.long()<<(torch.arange(8,device=cb.device)*4)).sum(-1).to(torch.uint32)
def decode_mb(t,layer,proj,e,N,K):
 pre=f'model.layers.{layer}.mlp.experts.arvq_{proj}_'
 a,b=unpack(t[pre+'packed'][e],N,K)
 cb=t[pre+'codebooks'][e].long();factors=t[pre+'book_factors'].float();B=len(factors)
 if cb.shape!=(256+B*256,):raise ValueError('Invalid mcbook packed codebook shape')
 values=torch.tensor(LEVELS,device=cb.device)[(cb[:,None]>>(torch.arange(8,device=cb.device)*4))&15]
 values=torch.cat([values[:256],values[256:]*factors.repeat_interleave(256)[:,None]])
 sel=t[pre+'selectors'][e].long()
 m=sel.repeat_interleave(16,0).repeat_interleave(8,1)
 stored=t[pre+'scales'][e].permute(0,2,1).contiguous().reshape(N,K//128)
 if stored.dtype==torch.float16:scales=stored.float()
 elif stored.dtype==torch.uint8:scales=stored.view(torch.float8_e4m3fn).float()
 else:raise ValueError('Unsupported packed block scale dtype')
 return ((values[a.long()]+values[256+m*256+b.long()])*scales.repeat_interleave(16,1)[...,None]*t[pre+'global']).reshape(N,K)
def export_layer_mb(source,dest,L):
 """mcbook16 export (format version 5): 16 residual books, per-tile 4-bit selectors,
 fp16 block scales; packed index stream and scales unchanged from v4."""
 from safetensors.torch import save_file,load_file
 source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 original=json.loads((source/'arvq-manifest.json').read_text());files={};first_factors=None
 for proj,tag in [('w13','gateup'),('w2','down')]:
  d=torch.load(source/f'{proj}.pt',map_location='cpu',weights_only=True)[proj];N,K=d['N'],d['K'];E=len(d['a']);pre=f'model.layers.{L}.mlp.experts.arvq_{proj}_'
  if d.get('scale_dtype')!='fp16':raise ValueError('mcbook export requires fp16 block scales')
  if not torch.equal(d['s'],d['s'].half().float()):raise ValueError('Scales are not FP16-representable')
  factors=[float(f) for f in d['book_factor']]
  if first_factors is None:first_factors=factors
  elif factors!=first_factors:raise ValueError('Mixed projection book factors')
  sel=d['selector']
  if sel.shape!=(E,N//16,K//64) or int(sel.max())>=len(factors):raise ValueError('Invalid selector tensor')
  t={pre+'packed':torch.stack([pack(a,b,N,K) for a,b in zip(d['a'],d['b'])]),
     pre+'codebooks':pack_codebooks_mb(d['cb'],factors),
     pre+'selectors':sel.to(torch.uint8).contiguous(),
     pre+'book_factors':torch.tensor(factors,dtype=torch.float32),
     pre+'scales':d['s'].half().reshape(E,N//16,16,K//128).permute(0,1,3,2).contiguous(),
     pre+'global':torch.tensor([d['glob']],dtype=torch.float32)}
  for e in range(E):
   a,b=unpack(t[pre+'packed'][e],N,K);assert torch.equal(a,d['a'][e]) and torch.equal(b,d['b'][e])
  for e in sorted({0,E//2,E-1}):
   m=d['selector'][e].long().repeat_interleave(16,0).repeat_interleave(8,1)
   cb=d['cb'][e]
   ref=((cb[d['a'][e].long()]+cb[256+m*256+d['b'][e].long()])*d['s'][e].repeat_interleave(16,1)[...,None]*d['glob']).reshape(N,K)
   assert torch.equal(ref,decode_mb(t,L,proj,e,N,K))
  name=f'arvq-layer-{L:03d}-{tag}.safetensors';tmp=dest/(name+'.tmp')
  save_file(t,str(tmp),metadata={'format':'pt','arvq_format':'rvq256_mb16_256x8_expert_fp16block','codebook_scope':'expert','residual_books':str(len(factors)),'book_factors':json.dumps(factors),'fit':str(original.get('fit','initial Hessian fit, no PV')),'serving_compatibility':'requires mcbook16 v5 loader/kernel (vllm-glm52-sm120 branch experiment/arvq-mcbook16)'})
  tmp.replace(dest/name)
  loaded=load_file(str(dest/name));assert loaded[pre+'packed'].shape==(E,N//16,K//64,64) and loaded[pre+'selectors'].shape==(E,N//16,K//64);del loaded
  files[proj]={'file':name,'sha256':sha(dest/name)};del t,d
 m={**original,'format':'rvq256_mb16_256x8_expert_fp16block','version':5,'residual_scale_shift':0,'block_scale_dtype':'float16','codebook_scope':'expert','bits':2.125+4/1024,'codebook_sizes':[256,16*256],'index_bits':[8,8],'selector_bits_per_tile':4,'words_per_tile':64,'book_factors':first_factors,'files':files,'fit':str(original.get('fit','initial Hessian fit, no PV')),'production_ready':False,'roundtrip':'all expert indices exact; 3 decoded weights/projection exact','requires_8x8_loader_kernel':True,'requires_mcbook16_loader':True}
 (dest/f'layer-{L:03d}-manifest.json').write_text(json.dumps(m,indent=2));return m
def decode(t,layer,proj,e,N=None,K=None,residual_scale_shift=0):
 pre=f'model.layers.{layer}.mlp.experts.arvq_{proj}_'
 default=(4096,6144) if proj=='w13' else (6144,2048)
 N,K=default if N is None else (N,K)
 a,b=unpack(t[pre+'packed'][e],N,K);cb=t[pre+'codebooks'].long()
 if cb.ndim==2:cb=cb[e]
 if cb.shape!=(512,):raise ValueError('Invalid packed codebook shape')
 values=torch.tensor(LEVELS,device=cb.device)[(cb[:,None]>>(torch.arange(8,device=cb.device)*4))&15]
 stored=t[pre+'scales'][e].permute(0,2,1).contiguous().reshape(N,K//128)
 if stored.dtype==torch.uint8:scales=stored.view(torch.float8_e4m3fn).float()
 elif stored.dtype==torch.float16:scales=stored.float()
 else:raise ValueError('Unsupported packed block scale dtype')
 f=2.0**(-int(residual_scale_shift))
 return ((values[a.long()]+f*values[256+b.long()])*scales.repeat_interleave(16,1)[...,None]*t[pre+'global']).reshape(N,K)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def export_layer(source,dest,L,residual_scale_shift=0):
 from safetensors.torch import save_file,load_file
 source=Path(source);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 probe=torch.load(source/'w13.pt',map_location='cpu',weights_only=True,mmap=True)['w13']
 if 'selector' in probe:
  del probe
  if int(residual_scale_shift):raise ValueError('mcbook exports are shift-0 (per-book factors)')
  return export_layer_mb(source,dest,L)
 del probe
 rss=int(residual_scale_shift);f=2.0**(-rss)
 original=json.loads((source/'arvq-manifest.json').read_text());files={}
 for proj,tag in [('w13','gateup'),('w2','down')]:
  d=torch.load(source/f'{proj}.pt',map_location='cpu',weights_only=True)[proj];N,K=d['N'],d['K'];E=len(d['a']);pre=f'model.layers.{L}.mlp.experts.arvq_{proj}_'
  expert=d['c0'].ndim==3
  if expert and d['c0'].shape!=(E,256,8):raise ValueError('Codebook expert count mismatch')
  if proj=='w13':scope='expert' if expert else 'layer'
  elif scope!=('expert' if expert else 'layer'):raise ValueError('Mixed projection codebook scopes')
  fp16=d.get('scale_dtype','fp8_e4m3')=='fp16'
  if proj=='w13':fp16_blocks=fp16
  elif fp16_blocks!=fp16:raise ValueError('Mixed projection scale precision')
  if fp16 and not torch.equal(d['s'],d['s'].half().float()):raise ValueError('Scales are not FP16-representable')
  stored_scales=d['s'].half() if fp16 else d['s'].to(torch.float8_e4m3fn).view(torch.uint8)
  t={pre+'packed':torch.stack([pack(a,b,N,K) for a,b in zip(d['a'],d['b'])]),
     pre+'codebooks':pack_codebooks(d['c0'],d['c1']),
     pre+'scales':stored_scales.reshape(E,N//16,16,K//128).permute(0,1,3,2).contiguous(),
     pre+'global':torch.tensor([d['glob']],dtype=torch.float32)}
  for e in range(E):
   a,b=unpack(t[pre+'packed'][e],N,K);assert torch.equal(a,d['a'][e]) and torch.equal(b,d['b'][e])
  for e in sorted({0,E//2,E-1}):
   c0,c1=(d['c0'][e],d['c1'][e]) if expert else (d['c0'],d['c1'])
   ref=((c0[d['a'][e].long()]+f*c1[d['b'][e].long()])*d['s'][e].repeat_interleave(16,1)[...,None]*d['glob']).reshape(N,K)
   assert torch.equal(ref,decode(t,L,proj,e,N,K,rss))
  base_fmt='rvq256_256x8_expert_fp16block' if fp16 else ('rvq256_256x8_expert' if expert else 'rvq256_256x8')
  arvq_fmt=base_fmt+(f'_rs{int(2**rss)}' if rss else '')
  name=f'arvq-layer-{L:03d}-{tag}.safetensors';tmp=dest/(name+'.tmp')
  save_file(t,str(tmp),metadata={'format':'pt','arvq_format':arvq_fmt,'residual_scale_shift':str(rss),'codebook_scope':scope,'fit':str(original.get('fit','initial Hessian fit, no PV')),'serving_compatibility':'requires FP16 block-scale v4 loader/kernel' if fp16 else ('requires per-expert v3 loader/kernel' if expert else 'requires 8+8 loader/kernel')})
  tmp.replace(dest/name);del t,d
  loaded=load_file(str(dest/name));assert loaded[pre+'packed'].shape==(E,N//16,K//64,64);del loaded
  files[proj]={'file':name,'sha256':sha(dest/name)}
 base_mfmt='rvq256_256x8_expert_fp16block' if fp16_blocks else ('rvq256_256x8_expert' if scope=='expert' else 'rvq256_256x8')
 m={**original,'format':base_mfmt+(f'_rs{int(2**rss)}' if rss else ''),'version':4 if fp16_blocks else (3 if scope=='expert' else 2),'residual_scale_shift':rss,'block_scale_dtype':'float16' if fp16_blocks else 'float8_e4m3fn','codebook_scope':scope,'bits':2.125 if fp16_blocks else 2.0625,'codebook_sizes':[256,256],'index_bits':[8,8],'words_per_tile':64,'files':files,'fit':str(original.get('fit','initial Hessian fit, no PV')),'production_ready':False,'roundtrip':'all expert indices exact; 3 decoded weights/projection exact','requires_8x8_loader_kernel':True}
 (dest/f'layer-{L:03d}-manifest.json').write_text(json.dumps(m,indent=2));return m
