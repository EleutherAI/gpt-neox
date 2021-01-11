"""
Benchmark performance of various datasets / loading.
Main purpose of this experiment is to test the best chunksize for dataset
and whether tokenizing has an effect on load time / file size
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from timeit import default_timer as timer


from gpt_neox.datasets import JsonShardedDataset
from gpt_neox.data_utils import shardify,  get_tokenizer, get_dir_size, remove_dir_files
import matplotlib.pyplot as plt

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

#helper for ignoring data we couldn't load
def ignore_exceptions_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

# Data parallel arguments.
batch_size=100
total_shards_to_load=1000
chunksizes=[10,100,1000]
num_workers = 32
initial_seed = 7
seq_length = 2048
data_path = ["/ml_data/the-pile/components/enron_emails/enron_emails.jsonl"]
output_dir = "/ml_data/test_output"
shard_meta_file_name = f"sharding.jsonl"



for pre_tokenize in [False,True]:
    load_time,shard_time,shard_sizes = [],[],[]
    for chunksize in chunksizes:
        start = timer()
        if pre_tokenize:
            tokenizer = get_tokenizer()
            shardify(data_path,shard_meta_file_name,seq_length,chunksize,output_dir,tokenizer=tokenizer)
        else:
            shardify(data_path,shard_meta_file_name,seq_length,chunksize,output_dir,tokenizer=None)
        dir_size = get_dir_size(output_dir)
        shard_sizes.append(dir_size)
        end = timer()
        elapsed = 1000*(end-start)
        shard_time.append(elapsed)
        if pre_tokenize:
            dataset = JsonShardedDataset(output_dir+"/"+shard_meta_file_name,None,seq_length,initial_seed)
        else:
            tokenizer = get_tokenizer()
            dataset = JsonShardedDataset(output_dir+"/"+shard_meta_file_name,tokenizer,seq_length,initial_seed)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,collate_fn=ignore_exceptions_collate)
        start = timer()
        #actually load all the entries as we would during training/testing
        for i in range(total_shards_to_load // batch_size):
            line = next(iter(dataloader))
            assert line is not None
        end = timer()
        elapsed = 1000*(end-start)
        load_time.append(elapsed)

        remove_dir_files(output_dir)

    if pre_tokenize:
        print("===pre-tokenizing===")
    else:
        print("===not pre-tokenizing===")
    print("Average sharding time: ",sum(shard_time)/len(shard_time), " ms")
    print("Average shard dir size: ",sum(shard_sizes)/len(shard_sizes), " bytes")
    print(f"Average loading time for {total_shards_to_load} seqs: ",sum(load_time)/len(load_time), " ms")
    
        
    plt.plot(chunksizes,load_time)
    plt.xlabel("Chunksize")
    plt.ylabel(f"Load time for {total_shards_to_load} seqs (ms)")
    plt.xscale("log")

    if pre_tokenize:
        plt.title("Pre-tokenized")
    else:
        plt.title("On-fly tokenize")
    plt.show()
    plt.close()