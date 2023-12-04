# Copyright (c) 2021, EleutherAI
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

from tools.datasets.corpora import prepare_dataset, DATA_DOWNLOADERS
import argparse

TOKENIZER_CHOICES = [
    "HFGPT2Tokenizer",
    "HFTokenizer",
    "GPT2BPETokenizer",
    "CharLevelTokenizer",
    "TiktokenTokenizer",
    "SPMTokenizer",
]
DATASET_CHOICES = [i for i in DATA_DOWNLOADERS.keys() if i != "pass"]


def get_args():
    parser = argparse.ArgumentParser(description="Download & preprocess neox datasets")
    parser.add_argument(
        "dataset",
        nargs="?",
        default="enwik8",
        help="name of dataset to download.",
        choices=DATASET_CHOICES,
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        default="GPT2BPETokenizer",
        choices=TOKENIZER_CHOICES,
        help=f'Type of tokenizer to use - choose from {", ".join(TOKENIZER_CHOICES)}',
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        default=None,
        help=f"Directory to which to download datasets / tokenizer "
        f"files - defaults to ./data",
    )
    parser.add_argument(
        "-v", "--vocab-file", default=None, help=f"Tokenizer vocab file (if required)"
    )
    parser.add_argument(
        "-m", "--merge-file", default=None, help=f"Tokenizer merge file (if required)"
    )
    parser.add_argument(
        "-f",
        "--force-redownload",
        dest="force_redownload",
        default=False,
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    prepare_dataset(
        dataset_name=args.dataset,
        tokenizer_type=args.tokenizer,
        data_dir=args.data_dir,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file,
        force_redownload=args.force_redownload,
    )
