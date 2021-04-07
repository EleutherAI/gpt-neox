"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC

from glob import glob
import os
import json
from collections import defaultdict
from pathlib import Path
import unicodedata
# from sampler import WeightedSampler
import argparse
from argparse import Namespace

def load_jsonl(input_path, quiet=True) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    if not quiet:
        print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def json_iterator(input_dir, text_key='text'):
    all_jsonls = glob(f'{input_dir}/*.jsonl') + glob(f'{input_dir}/*.json')
    for j in all_jsonls:
        data = load_jsonl(j)[:1000]
        for doc in data:
            yield doc[text_key]
    

def train_tokenizer(input_dir, save_path, tokenizer_type="BPE", vocab_size=52000):
    """
    Trains a tokenizer using all the languages contained in `langs` (a list of language ISO codes)

    :param input_dir:
    :param save_path:
    :param tokenizer_type:
    :param vocab_size:
    :return:
    """

    if tokenizer_type == "BPE":
        model = models.BPE()
    else:
        raise NotImplementedError(f'Tokenizer type {tokenizer_type} not implemented')
    tokenizer = Tokenizer(model)

    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.normalizer = NFKC()

    # And then train
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|padding|>"])
    tokenizer.train_from_iterator(json_iterator(input_dir), trainer)

    # And Save it
    tokenizer.save(save_path, pretty=True)
    print(f'Tokenizer saved at {save_path}')

def parse_args():
    parser = argparse.ArgumentParser(description='script for training a multilingual '
                                                 'HF tokenizer on CC dumps with upweighting for low resource languages')
    parser.add_argument('--json_input_dir', type=str,
                        help='Path to folder containing tokenizer training data in jsonl format')
    parser.add_argument('--tokenizer_output_path', type=str,
                        help='Path to which your trained tokenizer will be saved (should end in .json)')
    parser.add_argument('--tokenizer_type', type=str,
                        help="type of tokenizer to train, currently only BPE is supported",
                        choices=['BPE'],
                        default=['BPE'])
    parser.add_argument('-v', '--vocab_size',
                        help='vocabulary size of tokenizer, default=52k',
                        type=int, default=52000)
    return parser.parse_args()


if __name__ == "__main__":

    # args = parse_args()
    args = Namespace()
    args.json_input_dir = "/mnt/ssd-cluster/data/enron"
    args.tokenizer_type = "BPE"
    args.vocab_size = 52000
    args.tokenizer_output_path = "/mnt/ssd-cluster/data/tokenizer.json"

    train_tokenizer(args.json_input_dir,
                    save_path=args.tokenizer_output_path,
                    tokenizer_type=args.tokenizer_type,
                    vocab_size=args.vocab_size)
