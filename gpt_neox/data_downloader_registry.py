import os
import tarfile
from abc import ABC, abstractmethod
from glob import glob
import shutil
import random
import zstandard

"""
This registry is for automatically downloading and extracting datasets.

To register a class you need to inherit the DataDownloader class and provide name, filetype and url attributes, and 
(optionally) provide download / extract / exists functions to check if the data exists, and, if it doesn't, download and 
extract the data and move it to the correct directory.

When done, add it to the DATA_DOWNLOADERS dict. The function process_data runs the pre-processing for the selected 
dataset.
"""


class DataDownloader(ABC):
    """Dataset registry class to automatically download / extract datasets"""

    @property
    def base_dir(self):
        """base data directory"""
        return "./data"

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

    def _extract_zstd(self):
        self.path = os.path.join(self.base_dir, self.name)
        os.makedirs(self.path, exist_ok=True)
        zstd_file_path = os.path.join(self.base_dir, os.path.basename(self.url))
        with open(zstd_file_path, 'rb') as compressed:
            decomp = zstandard.ZstdDecompressor()
            output_path = zstd_file_path.replace(".zst", "")
            with open(output_path, 'wb') as destination:
                decomp.copy_stream(compressed, destination)

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

    def prepare(self):
        if not self.exists():
            self.download()
            self.extract()

class EnronJsonl(DataDownloader):
    name = "enron_jsonl"
    filetype = "jsonl.zst"
    url = "http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"
    seed = 1

    def exists(self):
        self.path = os.path.join(self.base_dir, self.name)
        return os.path.isfile(os.path.join(self.path, os.path.basename(self.url).replace(".zst", "")))

    def extract(self):
        self._extract_zstd()
        shutil.move(os.path.join(self.base_dir, os.path.basename(self.url).replace(".zst", "")), os.path.join(self.base_dir, self.name))

class EnronTFRecords(DataDownloader):
    name = "enron_tfr"
    filetype = "jsonl.zst"
    url = "http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"
    seed = 1

    def download(self):
        """downloads dataset"""
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, self.name)
        os.makedirs(self.path, exist_ok=True)
        os.system(f"wget {self.url} -O {os.path.join(self.path, os.path.basename(self.url))}")

    def extract(self):
        os.system(f"python ./gpt_neox/create_tfrecords.py --input_dir {self.path} --files_per 2000 \
        --name enron --output_dir {self.path}/tokenized --ftfy --chunk_size 1024 --processes 1")


class OWT2(DataDownloader):
    name = "owt2"
    filetype = "tfrecords"
    url = "http://eaidata.bmk.sh/data/owt2_new.tar.gz"
    seed = 1

    def extract(self):
        self._extract_tar()
        # the files are within nested subdirectories, and not split by train/test
        # so we need to move them to the correct directories
        all_files = glob(f"{self.path}/**/*.{self.filetype}", recursive=True)
        print(all_files)
        train_dir = f"{self.path}/train"
        eval_dir = f"{self.path}/eval"
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        total_tfrecords = len(all_files)
        n_eval_tfrecords = total_tfrecords // 10
        # owt2 doesn't have an official train/test split, so sample at random from tfrecords
        random.seed(self.seed)
        random.shuffle(all_files)
        eval_set = all_files[:n_eval_tfrecords]
        train_set = all_files[n_eval_tfrecords:]
        for f in train_set:
            shutil.move(f, train_dir)
        for f in eval_set:
            shutil.move(f, eval_dir)
        dirs_to_remove = [f for f in glob(f"{self.path}/*") if f not in [train_dir, eval_dir]]
        for d in dirs_to_remove:
            shutil.rmtree(d)


class Enwik8(DataDownloader):
    name = "owt2"
    filetype = "gz"
    url = "http://eaidata.bmk.sh/data/enwik8.gz"

    def extract(self):
        pass

    def exists(self):
        return os.path.isfile(f"{self.base_dir}/enwik8.gz")


DATA_DOWNLOADERS = {
    "owt2": OWT2,
    "enwik8": Enwik8,
    "enron_jsonl": EnronJsonl,
    "enron_tfr": EnronTFRecords
}


def prepare_data(dataset_name):
    DownloaderClass = DATA_DOWNLOADERS.get(dataset_name, None)
    if DownloaderClass is None:
        raise NotImplementedError
    else:
        d = DownloaderClass()
        d.prepare()
