import os
import tarfile
from abc import ABC, abstractmethod
from glob import glob
import shutil
import random
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
    def filetype(self):
        """filetype of dataset"""
        pass

    @property
    @abstractmethod
    def url(self):
        """URL from which to download dataset"""
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

    def _extract_tar(self):
        self.path = os.path.join(self.base_dir, self.name)
        os.makedirs(self.path, exist_ok=True)
        tarfile_path = os.path.join(self.base_dir, os.path.basename(self.url))
        with tarfile.open(tarfile_path, "r:gz") as dataset_tar:
            print(f'Extracting files from {tarfile_path}...')
            dataset_tar.extractall(self.path)

    def _extract_zstd(self, remove_zstd=True):
        self.path = os.path.join(self.base_dir, self.name)
        os.makedirs(self.path, exist_ok=True)
        zstd_file_path = os.path.join(self.base_dir, os.path.basename(self.url))
        with open(zstd_file_path, 'rb') as compressed:
            decomp = zstandard.ZstdDecompressor()
            output_path = zstd_file_path.replace(".zst", "")
            with open(output_path, 'wb') as destination:
                decomp.copy_stream(compressed, destination)
        if remove_zstd:
            os.remove(zstd_file_path)
        return output_path

    def extract(self):
        """extracts dataset and moves to the correct data dir if necessary"""
        self._extract_tar()

    def exists(self):
        """Checks if the dataset is present"""
        return os.path.isdir(f"{self.base_dir}/{self.name}")

    def download(self):
        """downloads dataset"""
        os.makedirs(self.base_dir, exist_ok=True)
        os.system(f"wget {self.url} -O {os.path.join(self.base_dir, os.path.basename(self.url))}")

    def tokenize(self):
        parent_folder = os.path.join(self.base_dir, self.name)
        jsonl_filepath = os.path.join(parent_folder, os.path.basename(self.url)).replace(".zst", "")
        assert jsonl_filepath.endswith(".jsonl")
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
            self.extract()
            self.tokenize()

class Enron(DataDownloader):
    name = "enron"
    filetype = "jsonl.zst"
    url = "http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"
    seed = 1

    def exists(self):
        self.path = os.path.join(self.base_dir, self.name)
        return os.path.isfile(os.path.join(self.path, os.path.basename(self.url).replace(".zst", "")))

    def extract(self, remove_zstd=True):
        self._extract_zstd(remove_zstd=remove_zstd)
        shutil.move(os.path.join(self.base_dir, os.path.basename(self.url).replace(".zst", "")), os.path.join(self.base_dir, self.name))

def maybe_download_gpt2_tokenizer_data(tokenizer_type):
    if tokenizer_type is None or tokenizer_type == DEFAULT_TOKENIZER_TYPE:
        if not os.path.isfile(GPT2_VOCAB_FP):
            os.system(f'wget {GPT2_VOCAB_URL} -O {GPT2_VOCAB_FP}')
        if not os.path.isfile(GPT2_MERGE_FP):
            os.system(f'wget {GPT2_MERGE_URL} -O {GPT2_MERGE_FP}')

DATA_DOWNLOADERS = {
    "enron": Enron
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
