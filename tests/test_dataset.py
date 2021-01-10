"""
Benchmark performance of various datasets / loading.
Main purpose of this experiment is to test the best chunksize for dataset
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from timeit import default_timer as timer


from gpt_neox.datasets import JsonShardedDataset
from gpt_neox.data_utils import shardify,  get_tokenizer
import matplotlib.pyplot as plt

#helper for ignoring data we couldn't load
def ignore_exceptions_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

# Data parallel arguments.
batch_size=100
total_shards_to_load=1000000
chunksizes=[100,1000,10000,100000,100000000]
num_workers = 32
load_time = []
initial_seed = 7
seq_length = 2048
data_path = ["/ml_data/the-pile/components/enron_emails/enron_emails.jsonl"]
shard_meta_file_name = f"/ml_data/the-pile/components/enron_emails/enron_email_sharding.jsonl"


tokenizer = get_tokenizer()

for chunksize in chunksizes:
    shardify(data_path,shard_meta_file_name,seq_length,chunksize)
    dataset = JsonShardedDataset(shard_meta_file_name,tokenizer,seq_length,initial_seed)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers,collate_fn=ignore_exceptions_collate)
    start = timer()
    for i in range(total_shards_to_load // batch_size):
        line = next(iter(dataloader))
        assert line is not None
    end = timer()
    elapsed = 1000*(end-start)
    print(elapsed)
    load_time.append(elapsed)
    
plt.plot(chunksizes,load_time)
plt.xlabel("Chunksize")
plt.ylabel(f"Load time for {total_shards_to_load} seqs (ms)")
plt.show()