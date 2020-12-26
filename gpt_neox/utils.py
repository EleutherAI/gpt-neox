import GPUtil
from threading import Thread
from tqdm.auto import tqdm
import time
import sys
import os
import numpy as np
import gzip
import torch


# helpers

def prepare_enwik8_data():
    if not os.path.isfile('./data/enwik8.gz'):
        os.system('wget http://eaidata.bmk.sh/data/enwik8.gz -O ./data/enwik8.gz')

    with gzip.open('./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)
    return data_train, data_val


def cycle(loader):
    while True:
        for data in loader:
            yield data


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))


def prepare_optimizer_parameters(model):
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    weight_decay = 0.01
    optimizer_grouped_parameters = [{
        'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
            weight_decay
    }, {
        'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
            0.0
    }]
    return optimizer_grouped_parameters

class DictArgs(dict):
    def __init__(self, config):
        for k, v in config.items():
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)
