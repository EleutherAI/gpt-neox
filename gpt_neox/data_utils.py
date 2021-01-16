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
from torch.utils.data.dataloader import default_collate
import pathlib
from functools import partial
import logging

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
    elif (tokenizer_type.lower() == "hf_gpt2tokenizer" and from_pretrained):
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

def get_dir_size(folder):
    files = os.listdir(folder)
    return sum([os.path.getsize(folder+"/"+f) for f in files])

def remove_dir_files(fdir):
    filelist = os.listdir(fdir) 
    for f in filelist:
        os.remove(os.path.join(fdir, f))

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

#helper for ignoring data we couldn't load
def ignore_exceptions_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    return default_collate(batch)