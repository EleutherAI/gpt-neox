#!/usr/bin/env python3
"""Build index-only homogeneous streams in a compute allocation, no token copies."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
c=json.loads(a.config.read_text()); root=a.output
if root.exists():
    raise ValueError('Refusing to overwrite a sampler')
root.mkdir(parents=True)
# Load only the mmap reader, without importing NeoX/distributed training.
source=a.repo/'megatron/data/indexed_dataset.py'
stub=types.ModuleType('megatron');stub.print_rank_0=print
sys.modules['megatron']=stub
spec=importlib.util.spec_from_file_location('indexed_dataset',source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
corpus=module.MMapIndexedDataset(c['data_path'],skip_warmup=True)
labels=np.load(c['gram_labels_path'],mmap_mode='r',allow_pickle=False)
if labels.shape!=corpus.sizes.shape or labels.dtype!=np.bool_:
    raise ValueError('Strong labels differ from the corpus')
labels=np.array(labels)
sizes_all=np.array(corpus.sizes)
print('Loaded document sizes and strong labels into RAM',flush=True)
samples=c['train_iters']*c['train_batch_size']; seq=c['seq_length']
totals=[0,0]; counts=[0,0]
chunk=1000000
for start in range(0,len(labels),chunk):
    flags=labels[start:start+chunk]; sizes=sizes_all[start:start+chunk]
    for route in (0,1):
        selected=sizes[flags==route]
        totals[route]+=int(selected.sum(dtype=np.int64)); counts[route]+=len(selected)
sample_counts=[round(samples*totals[0]/sum(totals)),0]
sample_counts[1]=samples-sample_counts[0]
epochs=max((sample_counts[i]*seq+1+totals[i]-1)//totals[i] for i in (0,1))
for route in (0,1):
    n=counts[route]*epochs
    # Random swapping against a Lustre-backed mmap causes millions of remote
    # page updates. Shuffle in RAM, then publish each array sequentially.
    ids=np.empty(n,dtype=np.int32)
    at=0
    for epoch in range(epochs):
        for start in range(0,len(labels),chunk):
            chosen=np.flatnonzero(labels[start:start+chunk]==route)+start
            ids[at:at+len(chosen)]=chosen;at+=len(chosen)
    np.random.RandomState(c['seed']+route).shuffle(ids)
    np.save(root/f'doc_ids_{route}.npy',ids)
    print('Shuffled and saved route',route,'documents',n,flush=True)
    offsets=np.empty(n+1,dtype=np.int64)
    offsets[0]=0
    for start in range(0,n,chunk):
        lengths=sizes_all[ids[start:start+chunk]]
        offsets[start+1:start+len(lengths)+1]=np.cumsum(lengths,dtype=np.int64)+offsets[start]
    np.save(root/f'offsets_{route}.npy',offsets)
    del ids,offsets
order=np.empty(samples,dtype=np.int64)
for start in range(0,samples,chunk):
    order[start:start+chunk]=np.arange(start,min(samples,start+chunk),dtype=np.int64)
np.random.RandomState(c['seed']+2).shuffle(order)
np.save(root/'sample_order.npy',order)
print('Saved shuffled sequence order',flush=True)
meta=dict(documents=len(labels),seq_length=seq,num_samples=samples,
          samples_per_route=sample_counts,corpus_tokens_per_route=totals,
          epochs=epochs,seed=c['seed'],data_path=c['data_path'],
          labels_path=c['gram_labels_path'],packing='separate clean/strong streams; proportional sample mix')
# Audit random sequences against original labels and source token slices.
source=a.repo/'megatron/data/gram_packed_dataset.py'
spec=importlib.util.spec_from_file_location('gram_packed_dataset',source)
packed=importlib.util.module_from_spec(spec);spec.loader.exec_module(packed)
# Construction is private until the compute job succeeds; publish only after audit.
(root/'BUILDING.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
    dataset=packed.HomogeneousPackedDataset(corpus,root,seq,samples,building=True)
    random=np.random.RandomState(42)
    for index in random.randint(0,samples,10000):
        sample=int(dataset.order[index]); route=int(sample>=sample_counts[0])
        local=sample-(sample_counts[0] if route else 0)
        ids,offsets=dataset.streams[route]
        first=int(np.searchsorted(offsets,local*seq,side='right')-1)
        last=int(np.searchsorted(offsets,local*seq+seq,side='right')-1)
        if not np.all(labels[ids[first:last+1]]==route):
            raise ValueError('Mixed-label sequence in homogeneous sampler')
        item=dataset[int(index)]
        if item['gram_flag']!=route or len(item['text'])!=seq+1:
            raise ValueError('Sampler text/label mismatch')
    meta['audit_sequences']=10000
    meta['mixed_label_sequences']=0
    (root/'COMPLETE.json').write_text(json.dumps(meta,indent=2)+'\n')
except BaseException:
    (root/'COMPLETE.json').unlink(missing_ok=True)
    raise
print(json.dumps(meta),flush=True)
