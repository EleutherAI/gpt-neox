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
from abc import ABC, abstractmethod
from multiprocessing import cpu_count

from .sglue_utils import get_super_glue_text_preprocessor

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
        num_workers=None,
        jsonl_keys=None,
    ):
        if tokenizer_type is None:
            tokenizer_type = "GPT2BPETokenizer"
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "./data")
        if merge_file is None:
            merge_file = f"{data_dir}/gpt2-merges.txt"
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
        self._num_workers = num_workers
        if jsonl_keys is None:
            self._jsonl_keys = ["text"]
        else:
            self._jsonl_keys = jsonl_keys

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

    @property
    def jsonl_keys(self):
        """keys to extract from jsonl"""
        return self._jsonl_keys

    def exists(self):
        """Checks if the dataset is present"""
        return os.path.isdir(f"{self.base_dir}/{self.name}")

    def download(self):
        """downloads dataset"""
        os.makedirs(os.path.join(self.base_dir, self.name), exist_ok=True)
        for url in self.urls:
            os.system(
                f"wget {url} -O {os.path.join(self.base_dir, self.name, os.path.basename(url))}"
            )

    def tokenize(self):
        """tokenizes dataset"""
        parent_folder = os.path.join(self.base_dir, self.name)
        jsonl_filepath = ",".join(
            [os.path.join(parent_folder, os.path.basename(url)) for url in self.urls]
        )

        cmd = f"python tools/preprocess_data.py \
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

        for key in self._jsonl_keys:
            _cmd = cmd + f"--jsonl-keys {key}"
            os.system(_cmd)

    def prepare(self):
        if not self.exists():
            self.download()
            self.tokenize()


class Enron(DataDownloader):
    name = "enron"
    urls = ["http://eaidata.bmk.sh/data/enron_emails.jsonl.zst"]
    num_docs = 517401


class PileSubset(DataDownloader):
    name = "pile_00"
    urls = ["https://mystic.the-eye.eu/public/AI/pile/train/00.jsonl.zst"]


class Pile(DataDownloader):
    name = "pile"
    urls = [
        f"https://mystic.the-eye.eu/public/AI/pile/train/{i:02}.jsonl.zst"
        for i in range(30)
    ]


class Github(DataDownloader):
    name = "github"
    urls = ["http://eaidata.bmk.sh/data/github_small.jsonl.zst"]


class ArXiv(DataDownloader):
    name = "arxiv"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/2020-09-08-arxiv-extracts-nofallback-until-2007-068.tar.gz"
    ]


class EuroParl(DataDownloader):
    name = "europarl"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/EuroParliamentProceedings_1996_2011.jsonl.zst"
    ]


class FreeLaw(DataDownloader):
    name = "freelaw"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst"
    ]


class NiH(DataDownloader):
    name = "nih"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst"
    ]


class PubMed(DataDownloader):
    name = "pubmed"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/PMC_extracts.tar.gz"
    ]


class Books1(DataDownloader):
    name = "books1"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/books1.tar.gz"
    ]


class Books3(DataDownloader):
    name = "books3"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/books3.tar.gz"
    ]


class HackerNews(DataDownloader):
    name = "hackernews"
    urls = ["https://mystic.the-eye.eu/public/AI/pile_preliminary_components/hn.tar.gz"]
    num_docs = 373000


class OpenWebText2(DataDownloader):
    name = "openwebtext2"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar"
    ]
    num_docs = 17103000


class StackExchange(DataDownloader):
    name = "stackexchange"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/stackexchange_dataset.tar"
    ]


class UbuntuIRC(DataDownloader):
    name = "ubuntu_irc"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/ubuntu_irc_until_2020_9_1.jsonl.zst"
    ]


class YoutubeSubtitles(DataDownloader):
    name = "youtube_subtitles"
    urls = [
        "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/yt_subs.jsonl.zst"
    ]


class C4(DataDownloader):
    name = "c4"
    urls = [
        f"https://the-eye.eu/public/AI/STAGING/c4/en/c4-train.{i:05}-of-01024.json.gz"
        for i in range(1024)
    ]


class C4OpenWebText(DataDownloader):
    name = "c4_openwebtext"
    urls = [
        f"https://mystic.the-eye.eu/eleuther_staging/c4/realnewslike/c4-train.{i:05}-of-00512.json.gz"
        for i in range(512)
    ]


class Enwik8(DataDownloader):
    name = "enwik8"
    urls = ["https://data.deepai.org/enwik8.zip"]


class SuperGLUE(DataDownloader):
    name = "super_glue"
    urls = [
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip",   # BoolQ
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/CB.zip",      # CB
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip",    # COPA
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip", # MultiRC
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",  # ReCoRD
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/RTE.zip",     # RTE
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip",     # WiC
        "https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip"      # WSC
        ]

    def _concat(self):
        import json

        dir_path = os.path.join(self.base_dir, self.name)
        for idx, url in enumerate(self.urls):
            os.system(
                f"unzip -o {os.path.join(dir_path, os.path.basename(url))} -d {dir_path}"
            )

            # \
            # f" -d {os.path.join(dir_path)}" \
            # f" && mv {os.path.join(dir_path, os.path.basename(url)[:-4])}/train.jsonl" \
            # f" {os.path.join(dir_path, os.path.basename(url))}.jsonl"
            file_dir = os.path.join(dir_path, os.path.basename(url)[:-4], "train.jsonl")

            preprocess_fn = get_super_glue_text_preprocessor(os.path.basename(url)[:-4])
            out_path = os.path.join(dir_path, os.path.basename(url)[:-4]+".jsonl")
            with open(file_dir, "r") as infile:
                with open(out_path, 'w') as outfile:
                    for line in infile:
                        preprocessed_line = preprocess_fn(json.loads(line))
                        if type(preprocessed_line) is list:
                            for _line in preprocessed_line:
                                json.dump(_line, outfile)
                                outfile.write('\n')
                        else:
                            json.dump(preprocessed_line, outfile)
                            outfile.write('\n')
                outfile.close()
            infile.close()

            self.urls[idx] = out_path

    def prepare(self):
        if not self.exists():
            self.download()
            self._concat()
            self.tokenize()


class P3(DataDownloader):
    name = "p3"
    urls = [] # download from a different script for now.
    
    def tokenize(self, filepath):
        """tokenizes dataset. override default cmd"""

        parent_folder = os.path.join(self.base_dir, self.name)
        
        # currently all content from different datasets is concatenated into one jsonl.
        # this is undesirable since we'd have to duplicate preproc for every version of p3 that we make.
        jsonl_filepath = ",".join(
            [os.path.join(parent_folder, (filepath))]
        )

        base_cmd = f"python tools/preprocess_data.py \
            --input {jsonl_filepath} \
            --vocab {self.vocab_file} \
            --dataset-impl mmap \
            --tokenizer-type {self.tokenizer_type} \
            --merge-file {self.merge_file} \
            --workers {self.num_workers} \
            --output-prefix {parent_folder}/{self.name} "

        if self.num_docs is not None:
            base_cmd += f"--num-docs {self.num_docs} "

        if self.ftfy:
            base_cmd += f"--ftfy "

        # only append EOD to the targets.
        for tokens_type in ["inputs", "targets"]:
            cmd = base_cmd
            if tokens_type == "targets": 
                cmd += "--append-eod "

            cmd += f"--jsonl-keys '{tokens_type}' "

            os.system(cmd)
    
    def prepare(self):
        self.name = "p3_train"
        self.download()
        self.tokenize("../p3_raw/p3_train.jsonl")
        self.name = "p3_valid"
        self.download()
        self.tokenize("../p3_raw/p3_validation.jsonl")


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
    "super_glue": SuperGLUE,
    "p3": P3,
}


def prepare_dataset(
    dataset_name: str,
    tokenizer_type: str = None,
    data_dir: str = None,
    vocab_file: str = None,
    merge_file: str = None,
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
        jsonl_keys = ["inputs", "targets"] if dataset_name == "super_glue" else ["text"]
        d = DownloaderClass(
            tokenizer_type=tokenizer_type,
            vocab_file=vocab_file,
            merge_file=merge_file,
            data_dir=data_dir,
            num_workers=num_workers,
            jsonl_keys=jsonl_keys,
        )
        d.prepare()
