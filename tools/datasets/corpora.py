# Copyright (c) 2021, EleutherAI
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
from abc import ABC, abstractmethod
from multiprocessing import cpu_count

"""
This registry is for automatically downloading and extracting datasets.

To register a class you need to inherit the DataDownloader class, and provide name and url attributes, and (optionally)
the number of documents.

When done, add it to the DATA_DOWNLOADERS dict. The function process_data runs the pre-processing for the selected
dataset.
"""

GPT2_VOCAB_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
GPT2_MERGE_URL = "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"


class DataDownloader(ABC):
    """Dataset registry class to automatically download / extract datasets"""

    def __init__(
        self,
        tokenizer_type=None,
        merge_file=None,
        vocab_file=None,
        data_dir=None,
        force_redownload=None,
        num_workers=None,
    ):
        if tokenizer_type is None:
            tokenizer_type = "GPT2BPETokenizer"
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "./data")
        if merge_file is None:
            merge_file = f"{data_dir}/gpt2-merges.txt"
        if force_redownload is None:
            force_redownload = False
        if vocab_file is None:
            if tokenizer_type == "GPT2BPETokenizer":
                vocab_file = f"{data_dir}/gpt2-vocab.json"
            elif tokenizer_type == "HFGPT2Tokenizer":
                vocab_file = "gpt2"
            elif tokenizer_type == "CharLevelTokenizer":
                pass
            else:
                assert vocab_file is not None, "No vocab file provided"
        if num_workers is None:
            num_workers = cpu_count()
        self._tokenizer_type = tokenizer_type
        self._merge_file = merge_file
        self._vocab_file = vocab_file
        self._data_dir = data_dir
        self._force_redownload = force_redownload
        self._num_workers = num_workers

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

    @property
    def num_workers(self):
        """Number of workers to use in preprocessing"""
        return self._num_workers

    @property
    def num_docs(self):
        """Number of documents in the dataset (if known)"""
        return None

    @property
    def ftfy(self):
        """Use ftfy (https://github.com/LuminosoInsight/python-ftfy) to fix text encodings"""
        return False

    def exists(self):
        """Checks if the dataset is present"""
        return os.path.isdir(f"{self.base_dir}/{self.name}")

    def download(self):
        """downloads dataset. In math-lm branch, this is replaced with a stub"""
        pass

    def tokenize(self):
        """tokenizes dataset"""
        parent_folder = os.path.join(self.base_dir, self.name)
        jsonl_filepath = ",".join(
            [
                os.path.join(parent_folder, f)
                for f in os.listdir(parent_folder)
                if f.endswith(".jsonl")
            ]
        )

        cmd = f"python tools/datasets/preprocess_data.py \
            --input {jsonl_filepath} \
            --output-prefix {parent_folder}/{self.name} \
            --vocab {self.vocab_file} \
            --dataset-impl mmap \
            --tokenizer-type {self.tokenizer_type} \
            --merge-file {self.merge_file} \
            --append-eod \
            --workers {self.num_workers} "

        if self.num_docs is not None:
            cmd += f"--num-docs {self.num_docs} "

        if self.ftfy:
            cmd += f"--ftfy "

        os.system(cmd)

    def prepare(self):
        self.tokenize()


class Enron(DataDownloader):
    name = "enron"
    urls = ["http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"]
    num_docs = 517401


class PileSubset(DataDownloader):
    name = "pile_00"
    urls = ["https://the-eye.eu/public/AI/pile/train/00.jsonl.zst"]


class Pile(DataDownloader):
    name = "pile"
    urls = [
        f"https://the-eye.eu/public/AI/pile/train/{i:02}.jsonl.zst" for i in range(30)
    ]


class Github(DataDownloader):
    name = "github"
    urls = ["http://eaidata.bmk.sh/data/github_small.jsonl.zst"]


class ArXiv(DataDownloader):
    name = "arxiv"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/2020-09-08-arxiv-extracts-nofallback-until-2007-068.tar.gz"
    ]


class EuroParl(DataDownloader):
    name = "europarl"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/EuroParliamentProceedings_1996_2011.jsonl.zst"
    ]


class FreeLaw(DataDownloader):
    name = "freelaw"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst"
    ]


class NiH(DataDownloader):
    name = "nih"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst"
    ]


class PubMed(DataDownloader):
    name = "pubmed"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/PMC_extracts.tar.gz"
    ]


class Books1(DataDownloader):
    name = "books1"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz"]


class Books3(DataDownloader):
    name = "books3"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/books3.tar.gz"]


class HackerNews(DataDownloader):
    name = "hackernews"
    urls = ["https://the-eye.eu/public/AI/pile_preliminary_components/hn.tar.gz"]
    num_docs = 373000


class OpenWebText2(DataDownloader):
    name = "openwebtext2"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar"
    ]
    num_docs = 17103000


class StackExchange(DataDownloader):
    name = "stackexchange"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/stackexchange_dataset.tar"
    ]


class UbuntuIRC(DataDownloader):
    name = "ubuntu_irc"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/ubuntu_irc_until_2020_9_1.jsonl.zst"
    ]


class YoutubeSubtitles(DataDownloader):
    name = "youtube_subtitles"
    urls = [
        "https://the-eye.eu/public/AI/pile_preliminary_components/yt_subs.jsonl.zst"
    ]


class C4(DataDownloader):
    name = "c4"
    urls = []


class C4OpenWebText(DataDownloader):
    name = "c4_openwebtext"
    urls = [
        f"https://the-eye.eu/eleuther_staging/c4/realnewslike/c4-train.{i:05}-of-00512.json.gz"
        for i in range(512)
    ]


class Enwik8(DataDownloader):
    name = "enwik8"
    urls = ["https://data.deepai.org/enwik8.zip"]

class ProofPile(DataDownloader): 
    name="proof-pile"
    urls=[]

class ArxivPilev2(DataDownloader):
    name="arxiv-pilev2"
    urls=[]

class ArxivRp(DataDownloader):
    name="arxiv-rp"
    urls=[]

class OpenWebMath(DataDownloader):
    name="open-web-math"
    urls=[]

class OpenWebMathHq(DataDownloader):
    name="open-web-math-hq"
    urls=[]

class ProofPilev1(DataDownloader):
    name="proof-pile-v1"
    urls=[]

class PileSample(DataDownloader):
    name="pile-sample"
    urls=[]

class PileSample1(DataDownloader):
    name="pile-sample-1"
    urls=[]

class PileSample2(DataDownloader):
    name="pile-sample-2"
    urls=[]

class CodeNoMatlab(DataDownloader):
    name="code-no-matlab"
    urls=[]

class CodeV1(DataDownloader):
    name="code-v1"
    urls=[]

class CodeV1WithTex(DataDownloader):
    name="code-v1-with-tex"
    urls=[]

class CodeV1FullMatlab(DataDownloader):
    name="code-v1-full-matlab"
    urls=[]

class CodeV2(DataDownloader):
    name="code-v2"
    urls=[]

class CodeV2WithTex(DataDownloader):
    name="code-v2-with-tex"
    urls=[]

class CodeV2FullMatlab(DataDownloader):
    name="code-v2-full-matlab"
    urls=[]

class CodeV3(DataDownloader):
    name="code-v3"
    urls=[]

class RedPajama(DataDownloader):
    name="red-pajama"
    urls=[]

def maybe_download_gpt2_tokenizer_data(tokenizer_type, data_dir):
    if tokenizer_type is None or tokenizer_type == "GPT2BPETokenizer":
        GPT2_VOCAB_FP = f"{data_dir}//gpt2-vocab.json"
        GPT2_MERGE_FP = f"{data_dir}/gpt2-merges.txt"
        if not os.path.isfile(GPT2_VOCAB_FP):
            os.system(f"wget {GPT2_VOCAB_URL} -O {GPT2_VOCAB_FP}")
        if not os.path.isfile(GPT2_MERGE_FP):
            os.system(f"wget {GPT2_MERGE_URL} -O {GPT2_MERGE_FP}")


DATA_DOWNLOADERS = {
    "pass": "pass",
    "enron": Enron,
    "pile_subset": PileSubset,
    "pile": Pile,
    "github": Github,
    "arxiv": ArXiv,
    "europarl": EuroParl,
    "freelaw": FreeLaw,
    "nih": NiH,
    "pubmed": PubMed,
    "books1": Books1,
    "books3": Books3,
    "hackernews": HackerNews,
    "openwebtext2": OpenWebText2,
    "stackexchange": StackExchange,
    "ubuntu_irc": UbuntuIRC,
    "youtube_subtitles": YoutubeSubtitles,
    "c4": C4,
    "c4_openwebtext": C4OpenWebText,
    "enwik8": Enwik8,
    "proof-pile": ProofPile,
    "arxiv-pilev2": ArxivPilev2,
    "arxiv-rp": ArxivRp,
    "open-web-math": OpenWebMath,
    "open-web-math-hq": OpenWebMathHq,
    "proof-pile-v1": ProofPilev1,
    "pile-sample": PileSample,
    "pile-sample-1": PileSample1,
    "pile-sample-2": PileSample2,
    "code-no-matlab": CodeNoMatlab,
    "code-v1": CodeV1,
    "code-v2": CodeV2,
    "code-v3": CodeV3,
    "code-v1-with-tex": CodeV1WithTex,
    "code-v2-with-tex": CodeV2WithTex,
    "code-v1-full-matlab": CodeV1FullMatlab,
    "code-v2-full-matlab": CodeV2FullMatlab,
    "red-pajama": RedPajama,
}


def prepare_dataset(
    dataset_name: str,
    tokenizer_type: str = None,
    data_dir: str = None,
    vocab_file: str = None,
    merge_file: str = None,
    force_redownload: bool = None,
    num_workers: int = None,
):
    """
    Downloads + tokenizes a dataset in the registry (dataset_name) and saves output .npy files to data_dir.
    """
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    maybe_download_gpt2_tokenizer_data(tokenizer_type, data_dir)
    DownloaderClass = DATA_DOWNLOADERS.get(dataset_name.lower(), None)
    if DownloaderClass is None:
        raise NotImplementedError(
            f'Dataset "{dataset_name}" not recognized - please choose from {list(DATA_DOWNLOADERS.keys())}'
        )
    elif DownloaderClass == "pass":
        # pass on building dataset (for unit tests)
        pass
    else:
        num_workers = 1 if dataset_name == "enwik8" else num_workers
        d = DownloaderClass(
            tokenizer_type=tokenizer_type,
            vocab_file=vocab_file,
            merge_file=merge_file,
            data_dir=data_dir,
            force_redownload=force_redownload,
            num_workers=num_workers,
        )
        d.prepare()
