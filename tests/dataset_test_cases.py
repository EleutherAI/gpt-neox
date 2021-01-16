import unittest
import os
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer


from gpt_neox.datasets import JsonShardedDataset
from gpt_neox.create_json_shards import shardify
from gpt_neox.data_utils import  get_tokenizer, get_dir_size, remove_dir_files, ignore_exceptions_collate
from gpt_neox.data_downloader_registry import prepare_data
import matplotlib.pyplot as plt
import linecache

batch_size=100
total_shards_to_load=10000
num_workers = 16
initial_seed = 7
seq_length = 2048
max_items_per_file = 100000
output_dir = "/ml_data/test_output"
shard_meta_file_name = f"sharding.jsonl"

dataset_names = ["enron_jsonl"]
data_paths = []
for name in dataset_names:
    file_path = prepare_data(name)
    data_paths.append(file_path)

def loading_pass(tokenizer,pretokenize):
    dir_size = get_dir_size(output_dir)
    print("DIR SIZE: ", dir_size)
    dataset = JsonShardedDataset(output_dir+"/"+shard_meta_file_name,tokenizer,seq_length,initial_seed,not pretokenize)
    
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                num_workers=num_workers,shuffle=True,collate_fn=ignore_exceptions_collate)
       
    #actually load all the entries as we would during training/testing
    iterator = iter(dataloader)
    loaded = 0
    expected_loaded = total_shards_to_load // batch_size
    for _ in range(expected_loaded):
        try:
            line = next(iterator)
            del line
            loaded += 1
        except:
            continue
    return loaded, expected_loaded

class TimedCase(unittest.TestCase):
    def setUp(self):
        self.start = timer()
    
    def tearDown(self):
        elapsed = timer() - self.start
        print(1000*(elapsed), " ms")

#exposes test cases for just the sharding process
class TestSharding(TimedCase):
    def test_pretokenized_sharding(self):
        print("=== Pretokenized sharding===")
        tokenizer = get_tokenizer()
        summary_dict = shardify(data_paths,shard_meta_file_name,max_items_per_file,output_dir,tokenizer=tokenizer,num_workers=num_workers)
        self.assertTrue(summary_dict)

    def test_not_pretokenized_sharding(self):
        print("=== Not pre-tokenized sharding===")
        summary_dict=shardify(data_paths,shard_meta_file_name,max_items_per_file,output_dir,tokenizer=None,num_workers=num_workers)
        self.assertTrue(summary_dict)

#exposes test cases for the sharding+loading process
class TestShardingAndLoading(unittest.TestCase):
    def test_pretokenized_zloading(self):
        print("=== Pretokenized sharding+loading pass===")
        tokenizer = get_tokenizer()
        summary_dict = shardify(data_paths,shard_meta_file_name,max_items_per_file,output_dir,tokenizer=tokenizer,num_workers=num_workers)
        start = timer()
        loaded, expected = loading_pass(tokenizer,True)
        end = timer()
        self.assertEqual(loaded,expected)
        print(f"Total time to load {total_shards_to_load} examples: {1000*(end-start)} ms")
    
    def test_not_pretokenized_zloading(self):
        print("=== Not pretokenized sharding+loading pass===")
        tokenizer = get_tokenizer()
        summary_dict = shardify(data_paths,shard_meta_file_name,max_items_per_file,output_dir,tokenizer=None,num_workers=num_workers)
        start = timer()
        loaded, expected = loading_pass(tokenizer,False)
        end = timer()
        self.assertEqual(loaded,expected)
        print(f"Total time to load {total_shards_to_load} examples: {1000*(end-start)} ms")

if __name__ == '__main__':
    unittest.main()
