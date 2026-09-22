"""Assemble a training batch once; distribute identical tensors to expert ranks."""
import torch
import torch.distributed as dist


def broadcast_batch(data,device):
    rank=dist.get_rank()
    # Fixed tensor schema, including explicit index dtype, shared by captures.
    keys=('x','topk_ids','topk_weights','required')
    if rank==0:
        shapes=torch.tensor([data['x'].shape[0],data['x'].shape[1],data['topk_ids'].shape[1]],dtype=torch.int64,device=device)
    else:shapes=torch.empty(3,dtype=torch.int64,device=device)
    dist.broadcast(shapes,0);rows,width,topk=shapes.tolist()
    result={}
    for key in keys:
        dtype=torch.int64 if key=='topk_ids' else torch.float32
        shape=(rows,topk if key in ('topk_ids','topk_weights') else width)
        result[key]=data[key].to(device=device,dtype=dtype).contiguous() if rank==0 else torch.empty(shape,device=device,dtype=dtype)
        dist.broadcast(result[key],0)
    has=torch.tensor(int("row_weight" in data) if rank==0 else 0,device=device)
    dist.broadcast(has,0)
    if int(has):
        weight=data["row_weight"].to(device=device,dtype=torch.float32).contiguous() if rank==0 else torch.empty(rows,device=device)
        dist.broadcast(weight,0);result["row_weight"]=weight
    return result
