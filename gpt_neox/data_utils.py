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
from multiprocessing import Process,Pool
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
Optionally shards data files into a certain size (determined as a multiple of number of entries),
then creates another "metadata" file that stores a single entry for every entry in the dataset,
so we can easily index into data for training

shards file is .jsonl with structure:
{seq_length:INT,total_lines:INT,total_shards:INT}\n
One line for each index
{file_name,line,start_index,end_index}
...
"""
def shardify(data_paths:list,output_path:str, seq_length:int=2048, chunksize:int=None, output_dir:str="",tokenizer=None,num_workers=128):
    summary_dict = {'seq_length':seq_length,'file_names':[]}
    
    shards = []
    total_shards,total_lines = 0, 0
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for path_idx,path in enumerate(data_paths):
        total_lines = file_lines(path)
        num_lines = total_lines//num_workers

        pool = Pool(processes = num_workers)
        s_process = partial(shardify_process,path=path,num_lines=num_lines,chunksize=chunksize,output_dir=output_dir,\
            tokenizer=tokenizer,seq_length=seq_length)
        returns = pool.map(s_process, range(num_workers))

        last_max_shard = 0
        for line_idx,single_file_shards,worker_shards,file_names in returns:
            total_lines += line_idx
            total_shards += single_file_shards
            
            max_ws = 0
            for ws in worker_shards:
                if ws[0] > max_ws:
                    max_ws = ws[0]
                ws[0]+=last_max_shard
            last_max_shard += max_ws
            shards.extend(worker_shards)
            summary_dict['file_names'].extend(file_names)
                
    summary_dict['total_lines'] = total_lines
    summary_dict['total_shards'] = total_shards

    with jsonlines.open(output_dir+"/"+output_path, mode='w') as writer:
        writer.write(summary_dict)
        for shard in shards:
            writer.write(shard)



def shardify_process(worker_id,path,num_lines,chunksize,output_dir,tokenizer,seq_length):
    start_line = worker_id*num_lines
    end_line = (worker_id+1)*num_lines

    ext_index = path.find(".")
    last_slash = path.rfind('/')
    single_file_shards,single_file_chunk,single_file_chunk_line,chunk_offset,chunk_shards = 0,0,0,0,0
        
    dataset_name = path[last_slash+1:ext_index]
    extension = path[ext_index:]

    file_names = []
    shards = []

    #if we specify a chunk size, we have to take the extra step of writing each chunk to a new file
    if chunksize:
        chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(0)+extension
        file_names.append(chunk_path)
        chunk_writer = jsonlines.Writer(open(chunk_path,"w"))
    else:
        file_names.append(path)

    with jsonlines.open(path) as reader:
        total_parsed = 0
        for line_idx,line_loaded in enumerate(reader):
            total_parsed += 1

            if line_idx<start_line or line_idx>end_line:
                continue
            
            text = line_loaded['text']
 
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
            line_shards=total_words // seq_length + 1 
            single_file_shards += line_shards
            for i in range(line_shards):
                #we've reached the end of this chunk, so need to reset everything for the next chunk
                if chunksize and single_file_chunk_line == chunksize:
                    chunk_path=output_dir+"/"+dataset_name+"_"+str(worker_id)+"_"+str(single_file_chunk)+extension
                    file_names.append(chunk_path)
                    chunk_writer = jsonlines.Writer(open(chunk_path,"w"))
                    single_file_chunk_line = 0
                    single_file_chunk += 1
                    chunk_offset += chunk_shards
                    chunk_shards = 0
           
                new_shard = []
                    
                if chunksize:
                    new_shard.append(single_file_chunk)
                    new_shard.append(single_file_chunk_line)
                    if tokenizer:
                        chunk_writer.write(all_words)
                    else:
                        chunk_writer.write(line_loaded)
                else:
                    new_shard.append(worker_id)
                    new_shard.append(worker_id)

                new_shard.append(seq_length * i - chunk_offset)
                end_index = seq_length * (i+1) - chunk_offset
                if end_index > total_words-chunk_offset:
                    end_index = total_words-chunk_offset
                new_shard.append(end_index)
                    
                shards.append(new_shard)
                single_file_chunk_line += 1

    return line_idx,single_file_shards,shards,file_names

def get_dir_size(folder):
    files = os.listdir(folder)
    return sum([os.path.getsize(folder+"/"+f) for f in files])

def remove_dir_files(fdir):
    filelist = os.listdir(fdir) 
    for f in filelist:
        os.remove(os.path.join(fdir, f))