"""Communicate only routed rows; reconstruct the exact dense proposal delta."""
import torch
import torch.distributed as dist

def sparse_broadcast_delta(delta,rows,owner):
    """Rows are unique affected indices, supplied only by owner.

Non-owner delta buffers are overwritten. The owning rank keeps its original
buffer, avoiding numerical changes in the candidate-loss calculation.
"""
    rank=dist.get_rank()
    n=torch.tensor(len(rows) if rank==owner else 0,device=delta.device,dtype=torch.int64)
    dist.broadcast(n,owner);count=int(n)
    if rank==owner:
        indices=rows.to(dtype=torch.int64).contiguous()
        payload=delta.index_select(0,indices).contiguous()
    else:
        indices=torch.empty(count,device=delta.device,dtype=torch.int64)
        payload=torch.empty((count,delta.shape[1]),device=delta.device,dtype=delta.dtype)
    if count:
        dist.broadcast(indices,owner);dist.broadcast(payload,owner)
    if rank!=owner:
        delta.zero_()
        if count:delta.index_copy_(0,indices,payload)
    return delta
