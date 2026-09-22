"""CPU-mapped training captures, with one shuffled no-replacement sequence pass."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch

class FullCorpus:
    def __init__(self,work,batch_tokens,seed=7103,layer=None):
        self.root=Path(work)/('training_capture' if layer is None else f'training_capture{layer}');self.locations={};self.shards={};self.coverage=torch.zeros(json.loads((Path(work)/"config.json").read_text())["num_experts"],dtype=torch.int64)
        self.rows=0;energy=0.;expected=None
        for rank in range(8):
            report=json.loads((self.root/f'rank{rank}_complete.json').read_text())
            if not report['complete']:raise ValueError('Capture incomplete')
            expected=report['total_corpus_tokens'];self.rows+=report['rows'];energy+=report['required_energy'];self.coverage+=torch.tensor(report['coverage'])
            for entry in report['entries']:
                path=self.root/entry['file'];self.shards[str(path)]=None;start=0
                for sequence,length in zip(entry['sequence_ids'],entry['sequence_lengths']):
                    if sequence in self.locations:raise ValueError('Duplicate captured sequence')
                    self.locations[sequence]=(str(path),start,start+length);start+=length
                if start!=entry['rows']:raise ValueError('Shard lengths mismatch')
        if self.rows!=expected or set(self.locations)!=set(range((expected+1023)//1024)):raise ValueError('Incomplete corpus coverage')
        if batch_tokens%1024:raise ValueError('Batch must contain whole sequences')
        self.order=np.random.default_rng(seed).permutation(len(self.locations)).tolist();self.sequences_per_batch=batch_tokens//1024
        self.steps=(len(self.order)+self.sequences_per_batch-1)//self.sequences_per_batch;self.denominator=energy/self.rows
        self.order_sha256=hashlib.sha256(np.array(self.order,dtype=np.int64).tobytes()).hexdigest()
    def get_sequences(self,sequences):
        chunks={k:[] for k in ['x','topk_ids','topk_weights','required']};rw=[];have_rw=None
        for seq in sequences:
            path,lo,hi=self.locations[seq]
            if self.shards[path] is None:self.shards[path]=torch.load(path,weights_only=True,mmap=True)
            shard=self.shards[path]
            for k in chunks:chunks[k].append(shard[k][lo:hi])
            if have_rw is None:have_rw='row_weight' in shard
            if have_rw:rw.append(shard['row_weight'][lo:hi])
        out={k:torch.cat(parts) for k,parts in chunks.items()}
        if rw:out['row_weight']=torch.cat(rw)
        return out
    def batch(self,step):
        lo=(step-1)*self.sequences_per_batch;seqs=self.order[lo:lo+self.sequences_per_batch]
        if not seqs:raise ValueError('No replacement: requested step beyond one corpus pass')
        return self.get_sequences(seqs)

    def batches(self,start=1):
        """One CPU batch ahead; deterministic ordering and bounded extra RAM."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.batch,start) if start<=self.steps else None
            for step in range(start,self.steps+1):
                batch=pending.result()
                pending=pool.submit(self.batch,step+1) if step<self.steps else None
                yield step,batch
