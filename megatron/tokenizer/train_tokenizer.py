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

"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from tokenizers.normalizers import NFKC

from glob import glob
import os
import json
import argparse


def load_jsonl(input_path, quiet=True) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    if not quiet:
        print("Loaded {} records from {}".format(len(data), input_path))
    return data


def json_iterator(input_dir, text_key="text"):
    all_jsonls = glob(f"{input_dir}/*.jsonl") + glob(f"{input_dir}/*.json")
    for j in all_jsonls:
        data = load_jsonl(j)
        for doc in data:
            yield doc[text_key]


def train_tokenizer(
    input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 52000
):
    """
    Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

    :param input_dir: input directory containing jsonl files
    :param save_path: path to save tokenizer to
    :param tokenizer_type: type of tokenizer to train.
    :param vocab_size: int, size of tokenizer's vocab
    :return:
    """

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f"Tokenizer type {tokenizer_type} not implemented")
    tokenizer = Tokenizer(model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # And then train
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|padding|>"]
    )
    tokenizer.train_from_iterator(json_iterator(input_dir), trainer)

    # And Save it
    tokenizer.save(save_path, pretty=True)
    print(f"Tokenizer saved at {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="script for training a multilingual "
        "HF tokenizer on CC dumps with upweighting for low resource languages"
    )
    parser.add_argument(
        "--json_input_dir",
        type=str,
        help="Path to folder containing tokenizer training data in jsonl format",
    )
    parser.add_argument(
        "--tokenizer_output_path",
        type=str,
        help="Path to which your trained tokenizer will be saved (should end in .json)",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        help="type of tokenizer to train, currently only BPE is supported",
        choices=["BPE"],
        default=["BPE"],
    )
    parser.add_argument(
        "-v",
        "--vocab_size",
        help="vocabulary size of tokenizer, default=52k",
        type=int,
        default=52000,
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_tokenizer(
        args.json_input_dir,
        save_path=args.tokenizer_output_path,
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
    )
