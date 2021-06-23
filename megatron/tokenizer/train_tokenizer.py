"""
Assumes a dataset of jsonl files in the same format as the neox training set.
"""

from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC

from glob import glob
import os
import json
import argparse
import lm_dataformat as lmd 
import random
from tqdm import tqdm
from multiprocessing import cpu_count
from functools import partial 

def json_iterator(input_dir, threaded=False, max_docs=None, max_sentence_length=None, split_by_newline=True):
    n = 0
    for f in filter(lambda x: x, tqdm(lmd.Reader(input_dir).stream_data(threaded=threaded))):
        if split_by_newline:
            for s in f.split('\n'):
                if max_sentence_length:
                    s = s[:max_sentence_length - 1]
                if s.strip():
                    yield s
        else:
            if max_sentence_length:
                f = f[:max_sentence_length - 1]
            yield f
        if max_docs is not None and n >= max_docs:
            break
        n += 1
    
class SPMTrainer:

    def __init__(self, vocab_size, model_type="bpe", special_tokens=None, split_by_whitespace=False, remove_extra_whitespaces=False, byte_fallback=True, split_digits=True):
        import sentencepiece as spm
        self.trainer = spm.SentencePieceTrainer
        self.pad_token = "<|padding|>"
        self.eos_token = "<|endoftext|>"
        self.bos_id = -1
        self.pad_id = 1
        if special_tokens is None:
            special_tokens = ["▁▁", "▁▁▁▁", "<|newline|>"]
        self.special_tokens = special_tokens
        self.split_by_whitespace = split_by_whitespace
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.vocab_size = vocab_size
        self.model_type = model_type
        assert self.model_type in ["bpe", "unigram"]
        self.byte_fallback = byte_fallback
        self.split_digits = split_digits
        self.processes = cpu_count()

    def train(self, iterator, model_prefix):
        self.trainer.train(sentence_iterator=iterator, model_prefix=model_prefix, vocab_size=self.vocab_size, model_type=self.model_type, user_defined_symbols=self.special_tokens, 
                                split_by_whitespace=self.split_by_whitespace, remove_extra_whitespaces=self.remove_extra_whitespaces, num_threads=self.processes, 
                                add_dummy_prefix=False, eos_piece=self.eos_token, pad_piece=self.pad_token, bos_id=self.bos_id, split_digits=self.split_digits, pad_id=self.pad_id)

def train_tokenizer(input_dir: str, save_path: str, tokenizer_type: str = "BPE", vocab_size: int = 52000, max_docs=2000000):
    """
    Trains a tokenizer on all the json files in `input_dir` and saves it to `save_path`

    :param input_dir: input directory containing jsonl files
    :param save_path: path to save tokenizer to
    :param tokenizer_type: type of tokenizer to train.
    :param vocab_size: int, size of tokenizer's vocab
    :return:
    """
    if tokenizer_type == "sp_bpe":
        trainer = SPMTrainer(vocab_size=vocab_size, model_type="bpe")
        trainer.train(json_iterator(input_dir, max_sentence_length=4192, max_docs=max_docs), save_path)
    elif tokenizer_type == "sp_unigram":
        trainer = SPMTrainer(vocab_size=vocab_size, model_type="unigram")
        trainer.train(json_iterator(input_dir, max_sentence_length=4192, max_docs=max_docs), save_path)
    elif tokenizer_type == "BPE":
        model = models.BPE()
        # Customize pre-tokenization and decoding
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", add_prefix_space=True)
        tokenizer.decoder = decoders.Metaspace(replacement="▁", add_prefix_space=True)
        # And then train
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>", "<|padding|>", " ", "  ", "    "])

        tokenizer.train_from_iterator(json_iterator(input_dir, max_docs=max_docs), trainer, length=length)

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
                        choices=['sp_bpe', 'sp_unigram', 'bpe'],
                        default='bpe')
    parser.add_argument('-v', '--vocab_size',
                        help='vocabulary size of tokenizer, default=52k',
                        type=int, default=52000)
    parser.add_argument('-m', '--max_docs',
                        help='Maximum number of documents to sample from in training',
                        default=2000000)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    train_tokenizer(args.json_input_dir,
                    save_path=args.tokenizer_output_path,
                    tokenizer_type=args.tokenizer_type,
                    vocab_size=args.vocab_size,
                    max_docs=args.max_docs)
