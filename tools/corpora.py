# coding=utf-8
# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import tarfile
from abc import ABC, abstractmethod
import shutil
import zstandard

"""
This registry is for automatically downloading and extracting datasets.
To register a class you need to inherit the DataDownloader class, provide name, filetype and url attributes, and 
(optionally) provide download / extract / exists / tokenize functions to check if the data exists, and, if it doesn't, download, 
extract and tokenize the data into the correct directory.
When done, add it to the DATA_DOWNLOADERS dict. The function process_data runs the pre-processing for the selected 
dataset.
"""

DEFAULT_DATA_DIR = os.environ.get('DATA_DIR', './data')

DEFAULT_TOKENIZER_TYPE = "GPT2BPETokenizer"
GPT2_VOCAB_FP = f"{DEFAULT_DATA_DIR}/gpt2-vocab.json"
GPT2_VOCAB_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
GPT2_MERGE_FP = f"{DEFAULT_DATA_DIR}/gpt2-merges.txt"
GPT2_MERGE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"


class DataDownloader(ABC):
    """Dataset registry class to automatically download / extract datasets"""

    def __init__(self, tokenizer_type=None, merge_file=None, vocab_file=None, data_dir=None):
        if tokenizer_type is None:
            tokenizer_type = DEFAULT_TOKENIZER_TYPE
        if merge_file is None:
            merge_file = GPT2_MERGE_FP
        if vocab_file is None:
            if tokenizer_type == DEFAULT_TOKENIZER_TYPE:
                vocab_file = GPT2_VOCAB_FP
            elif tokenizer_type == "HFGPT2Tokenizer":
                vocab_file = 'gpt2'
            else:
                assert vocab_file is not None, 'No vocab file provided'
        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self._tokenizer_type = tokenizer_type
        self._merge_file = merge_file
        self._vocab_file = vocab_file
        self._data_dir = data_dir

    @property
    def base_dir(self):
        """base data directory"""
        return self._data_dir

    @property
    @abstractmethod
    def name(self):
        """name of dataset"""
        pass

    @property
    @abstractmethod
    def urls(self):
        """URLs from which to download dataset"""
        pass

    @property
    def tokenizer_type(self):
        """tokenizer type to use when tokenizing data"""
        return self._tokenizer_type

    @property
    def merge_file(self):
        """Merge file for tokenizer"""
        return self._merge_file

    @property
    def vocab_file(self):
        """Vocab file for tokenizer"""
        return self._vocab_file

    def exists(self):
        """Checks if the dataset is present"""
        return os.path.isdir(f"{self.base_dir}/{self.name}")

    def download(self):
        """downloads dataset"""
        for url in self.urls:
            os.makedirs(self.base_dir, exist_ok=True)
            os.system(f"mkdir -p {os.path.join(self.base_dir, self.name)}")
            os.system(f"wget {url} -O {os.path.join(self.base_dir, self.name, os.path.basename(url))}")

    def tokenize(self):
        parent_folder = os.path.join(self.base_dir, self.name)
        jsonl_filepath = ",".join([
            os.path.join(parent_folder, os.path.basename(url)) 
            for url in self.urls
        ])
    
        os.system(f"python tools/preprocess_data.py \
            --input {jsonl_filepath} \
            --output-prefix {parent_folder}/{self.name} \
            --vocab {self.vocab_file} \
            --dataset-impl mmap \
            --tokenizer-type {self.tokenizer_type} \
            --merge-file {self.merge_file} \
            --append-eod")

    def prepare(self):
        if not self.exists():
            self.download()
            self.tokenize()


class Enron(DataDownloader):
    name = "enron"
    urls = ["http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"]


class PileSubset(DataDownloader):
    name = "pile_00"
    urls = ["https://the-eye.eu/public/AI/pile/train/00.jsonl.zst"]


class Pile(DataDownloader):
    name = "pile"
    urls = [f"https://the-eye.eu/public/AI/pile/train/{i:02}.jsonl.zst" for i in range(30)]


class Github(DataDownloader):
    name = "github"
    urls = ["http://eaidata.bmk.sh/data/github_small.jsonl.zst"]


class ArXiv(DataDownloader):
    name = "arxiv"
    urls = ["http://eaidata.bmk.sh/data/arxiv.jsonl.zst"]



def maybe_download_gpt2_tokenizer_data(tokenizer_type):
    if tokenizer_type is None or tokenizer_type == DEFAULT_TOKENIZER_TYPE:
        if not os.path.isfile(GPT2_VOCAB_FP):
            os.system(f'wget {GPT2_VOCAB_URL} -O {GPT2_VOCAB_FP}')
        if not os.path.isfile(GPT2_MERGE_FP):
            os.system(f'wget {GPT2_MERGE_URL} -O {GPT2_MERGE_FP}')


DATA_DOWNLOADERS = {
    "enron": Enron,
    "pile_subset": PileSubset,
    "pile": Pile,
    "github": Github,
    "arxiv": ArXiv,
}

def prepare_dataset(dataset_name: str, tokenizer_type: str = None, data_dir: str = None, vocab_file: str = None, merge_file: str = None):
    """
    Downloads + tokenizes a dataset in the registry (dataset_name) and saves output .npy files to data_dir.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    maybe_download_gpt2_tokenizer_data(tokenizer_type)
    DownloaderClass = DATA_DOWNLOADERS.get(dataset_name, None)
    if DownloaderClass is None:
        raise NotImplementedError
    else:
        d = DownloaderClass(tokenizer_type=tokenizer_type, vocab_file=vocab_file, merge_file=merge_file, data_dir=data_dir)
        d.prepare()
