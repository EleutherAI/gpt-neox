from transformers import GPT2TokenizerFast, GPT2Tokenizer
from itertools import islice
import re
import os
from collections import OrderedDict
import gzip
import numpy as np
import torch
import linecache
import jsonlines
import math
from multiprocessing import Process,Pool
import pathlib
from functools import partial


class FixedSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


def skip(iterator, n):
    return islice(iterator, n, None)


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def read_enwik8_data(data_path):
    with gzip.open(data_path) as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
    return data_train, data_val

def get_tokenizer(tokenizer_type=None, from_pretrained=True, add_padding_token=True):
    if tokenizer_type is None or (tokenizer_type.lower() == "hf_gpt2tokenizerfast" and from_pretrained):
        tok = GPT2TokenizerFast.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    elif tokenizer_type.lower() == "hf_gp2tokenizer" and from_pretrained:
        tok = GPT2Tokenizer.from_pretrained('gpt2')
        if add_padding_token:
            tok.add_special_tokens({'pad_token': '<|padding|>'})
        return tok
    else:
        raise NotImplementedError('TODO: add custom tokenizers')


def file_lines(fname):
    total_lines = 0
    with open(fname) as f:
        for l in f:
            total_lines +=1
    return total_lines + 1

"""
Shards data files into a certain size,
then creates another "metadata" file that stores a single entry for every entry in the dataset
so we can easily index into data for training

metadata file is .jsonl with structure:
{seq_length:INT,total_lines:INT,total_shards:INT}\n
One line for each index
{file_name,line}
...
"""
def shardify(data_paths:list,output_path:str, seq_length:int=2048, chunksize:int=None, output_dir:str="",tokenizer=None,num_workers=32):
    summary_dict = {'seq_length':seq_length,'file_names':[]}
    
    shards = []
    total_shards,total_lines = 0, 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    #splitting workers per path intead of giving each worker a different path
    for path_idx,path in enumerate(data_paths):
        total_lines = file_lines(path)
        num_lines = total_lines//num_workers

        pool = Pool(processes = num_workers)
        s_process = partial(shardify_process,path=path,num_lines=num_lines,chunksize=chunksize,output_dir=output_dir,\
            tokenizer=tokenizer,seq_length=seq_length)
        returns = pool.map(s_process, range(num_workers))

        for line_idx,path_shards,worker_shards,file_names in returns:
            total_lines += line_idx
            total_shards += path_shards
            
            max_ws = len(summary_dict['file_names'])
            for ws in worker_shards:
                ws[0]+=max_ws

            shards.extend(worker_shards)
            summary_dict['file_names'].extend(file_names)
                
    summary_dict['total_lines'] = total_lines
    summary_dict['total_shards'] = total_shards

    #finish by writing the summary dict and all the individual indexes
    with jsonlines.open(output_dir+"/"+output_path, mode='w') as writer:
        writer.write(summary_dict)
        for shard in shards:
            writer.write(shard)

#Runs on a single worker to chunk and optionally tokenize a jsonl file
def shardify_process(worker_id,path,num_lines,chunksize,output_dir,tokenizer,seq_length):
    path_shards,single_file_chunk,single_file_chunk_line = 0,1,0
    start_line = worker_id*num_lines
    end_line = (worker_id+1)*num_lines

    p = pathlib.Path(path)
    path_folder = p.parents[0]  
    dataset_name = p.stem
    extension = p.suffix

    file_names = []
    shards = []

    chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(0)+extension
    chunk_writer = None

    with jsonlines.open(path) as reader:
        total_parsed = 0
        for line_idx,line_loaded in enumerate(reader):
            #only have the worker process the lines of the jsonl that it has been assigned to
            if line_idx<start_line or line_idx>end_line:
                continue
            
            text = line_loaded['text']
            if len(text) == 0:
                continue
            total_parsed += 1

            if single_file_chunk_line == 0:
                file_names.append(chunk_path)
                if chunk_writer:
                    chunk_writer.close()
                chunk_writer = jsonlines.Writer(open(chunk_path,"w"))
 
            if tokenizer:
                all_words = tokenizer(text, max_length=seq_length, return_tensors='pt',\
                    truncation=True)['input_ids']
                all_words = all_words.numpy().tolist()[0]
                all_words = list(map(lambda x:int(x), all_words))
                
                total_words = len(all_words)
                
            else:
                all_words = text.split(" ")
                total_words = len(all_words)

            #individual shards is limited by context
            line_shards=math.floor(total_words / seq_length) + 1 
 
            path_shards += line_shards

            for i in range(line_shards):
                #each shard contains 
                # 1.what file we save it to (index to file_names array in summary)
                # 2.what line of the file it maps to
                new_shard = [single_file_chunk,single_file_chunk_line+1]

                start = i*seq_length
                stop = (i+1)*seq_length 
                trunc_words = all_words[start:stop]

                if len(trunc_words) == 0:
                    continue
                else:
                    chunk_writer.write(trunc_words)
                    shards.append(new_shard)
                    single_file_chunk_line += 1

                #we've reached the end of this chunk, so need to reset everything for the next chunk
                if single_file_chunk_line == chunksize:
                    chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(single_file_chunk)+extension
                    single_file_chunk_line = 0
                    single_file_chunk += 1
                

    #line_idx ends up being the number of lines processed
    #path shards is the total number of entries 
    #shards is a list of identifying info for each index consisting of
    return line_idx,path_shards,shards,file_names

def get_dir_size(folder):
    files = os.listdir(folder)
    return sum([os.path.getsize(folder+"/"+f) for f in files])

def remove_dir_files(fdir):
    filelist = os.listdir(fdir) 
    for f in filelist:
        os.remove(os.path.join(fdir, f))