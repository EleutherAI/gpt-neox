from transformers import GPT2TokenizerFast, GPT2Tokenizer
from itertools import islice
import re
from collections import OrderedDict
import gzip
import numpy as np
import torch
import linecache
import jsonlines

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
def shardify(data_paths:list,output_path:str, seq_length=2048, chunksize=None):
    summary_dict = {'seq_length':seq_length}
    shards = []
    total_shards,total_lines = 0, 0
    chunk_writer = None
    for path in data_paths:
        ext_index = path.find(".")
        single_file_lines, single_file_shards,single_file_chunk,single_file_chunk_line,chunk_offset,chunk_shards = 0,0,0,0,0,0
        #if we specify a chunk size, we have to take the extra step of writing each chunk to a new file
        if chunksize:
            chunk_path=path[:ext_index]+"_"+str(0)+path[ext_index:]
            chunk_writer = jsonlines.Writer(open(chunk_path,"w"))

        with jsonlines.open(path) as reader:
            for line_loaded in reader:
                
                text = line_loaded['text']
                single_file_lines+=1 
                all_words = text.split(" ")
                total_words = len(all_words)

                #individual shards is limited by context
                line_shards=total_words // seq_length + 1 
                single_file_shards += line_shards
                for i in range(line_shards):
                    #we've reached the end of this chunk, so need to reset everything for the next chunk
                    if chunksize and single_file_chunk_line == chunksize:
                        chunk_path=path[:ext_index]+"_"+str(single_file_chunk)+path[ext_index:]
                        chunk_writer = jsonlines.Writer(open(chunk_path,"w"))
                        single_file_chunk_line = 0
                        single_file_chunk += 1
                        chunk_offset += chunk_shards
                        chunk_shards = 0
                    new_shard = {}
                    
                    if chunksize:
                        new_shard['file_name'] = chunk_path
                        new_shard['line'] = single_file_chunk_line
                        chunk_writer.write(line_loaded)
                    else:
                        new_shard['file_name'] = path
                        new_shard['line'] = single_file_lines

                    new_shard['start_index'] = seq_length * i - chunk_offset
                    end_index = seq_length * (i+1) - chunk_offset
                    if end_index > total_words-chunk_offset:
                        end_index = total_words-chunk_offset
                    new_shard['end_index'] = end_index

                    shards.append(new_shard)
                    single_file_chunk_line += 1
  
        total_lines += single_file_lines
        total_shards += single_file_shards
                
    summary_dict['total_lines'] = total_lines
    summary_dict['total_shards'] = total_shards


    with jsonlines.open(output_path, mode='w') as writer:
        writer.write(summary_dict)
        for shard in shards:
            writer.write(shard)
