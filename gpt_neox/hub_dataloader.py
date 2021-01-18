import torch
from torch.utils.data import Dataset
from .data_utils import get_tokenizer, natural_sort, skip, FixedSizeOrderedDict
import random
import glob
import tensorflow as tf
import re
import logging
from itertools import cycle
import os
import subprocess
import simdjson as json
import hub

class HubAdapter(torch.utils.data.Dataset):
    def __init__(self, ods):
        self.ds = ods

    @classmethod
    def __instancecheck__(cls, instance):
        return isinstance(instance, torch.utils.data.Dataset)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        x = self.ds.__getitem__(index)
        return x['text']


def get_hub_dataset():
    schema = hub.schema.SchemaDict({'text': hub.schema.Tensor(shape=(None,), dtype='int64', max_shape=(2049,))})
    ds = hub.Dataset("snsi/pile_dev", schema=schema, shape=(100000,)).to_pytorch()
    return HubAdapter(ds)