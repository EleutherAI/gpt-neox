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

DATA_DIR = os.environ.get('DATA_DIR', './data')

GPT2_VOCAB_FP = f"{DATA_DIR}/gpt2-vocab.json"
GPT2_VOCAB_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
GPT2_MERGE_FP = f"{DATA_DIR}/gpt2-merges.txt"
GPT2_MERGE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"

class DataDownloader(ABC):
    """Dataset registry class to automatically download / extract datasets"""

    @property
    def base_dir(self):
        """base data directory"""
        return DATA_DIR

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
            --vocab {GPT2_VOCAB_FP} \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --merge-file {GPT2_MERGE_FP} \
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

def maybe_download_gpt2_tokenizer_data():
    if not os.path.isfile(GPT2_VOCAB_FP):
        os.system(f'wget {GPT2_VOCAB_URL} -O {GPT2_VOCAB_FP}')
    if not os.path.isfile(GPT2_MERGE_FP):
        os.system(f'wget {GPT2_MERGE_URL} -O {GPT2_MERGE_FP}')

DATA_DOWNLOADERS = {
    "enron": Enron
}

def prepare_dataset(dataset_name):
    os.makedirs(DATA_DIR, exist_ok=True)
    maybe_download_gpt2_tokenizer_data()
    DownloaderClass = DATA_DOWNLOADERS.get(dataset_name, None)
    if DownloaderClass is None:
        raise NotImplementedError
    else:
        d = DownloaderClass()
        d.prepare()
