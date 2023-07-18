# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod
import torch

from tokenizers import Tokenizer
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import numpy as np
import sentencepiece as spm
from typing import List, Union
from .gpt2_tokenization import GPT2Tokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type.lower() == "GPT2BPETokenizer".lower():
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type.lower() == "SPMTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = SentencePieceTokenizer(args.vocab_file)
    elif args.tokenizer_type.lower() == "HFTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = HFTokenizer(args.vocab_file,seq_length=args.seq_length)
    elif args.tokenizer_type.lower() == "HFGPT2Tokenizer".lower():
        tokenizer = HFGPT2Tokenizer(seq_length=args.seq_length)
    elif args.tokenizer_type.lower() == "CharLevelTokenizer".lower():
        tokenizer = CharLevelTokenizer(vocab_size=512)
    elif args.tokenizer_type.lower() == "TiktokenTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = TiktokenTokenizer(args.vocab_file)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = "GPT2 BPE"
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(
            vocab_file, merge_file, errors="replace", special_tokens=[], max_len=None
        )
        self.eod_id = self.tokenizer.encoder["<|endoftext|>"]

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class SentencePieceTokenizer(AbstractTokenizer):
    """Designed to Integrate SP's Tokenizer."""

    def __init__(self, vocab_file):
        name = "SPM"
        super().__init__(name)

        self.tokenizer = spm.SentencePieceProcessor(model_file=vocab_file)
        self.eod_id = self.tokenizer.piece_to_id("<|endoftext|>")

    @property
    def vocab_size(self):
        return self.tokenizer.get_piece_size()

    @property
    def vocab(self):
        return {
            self.tokenizer.id_to_piece(idx): idx
            for idx in range(self.tokenizer.get_piece_size())
        }

    @property
    def inv_vocab(self):
        return {
            idx: self.tokenizer.id_to_piece(idx)
            for idx in range(self.tokenizer.get_piece_size())
        }

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class HFTokenizer(AbstractTokenizer):
    """Designed to Integrate HF's Tokenizer library."""

    def __init__(self, vocab_file,seq_length):
        name = "HFTokenizer"
        super().__init__(name)
        print(vocab_file)
        self.tokenizer = Tokenizer.from_file(vocab_file)
        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")
        self.seq_length = seq_length
        self.tokenizer.enable_truncation(max_length=seq_length)

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, texts: Union[str, List[str]], context_length=2048):
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = [encoding.ids for encoding in self.tokenizer.encode_batch(texts)]
        # add eod_id and pad with pad_id
        for idx,ids in enumerate(input_ids):
            if len(ids) < self.seq_length:
                ids = ids+[self.eod_id]+[self.pad_id]*(self.seq_length-len(ids)-1)
            else: # truncated
                ids = ids[:-1]+[self.eod_id]
            input_ids[idx]=ids
        input_ids = torch.tensor(input_ids,dtype=torch.int64) 
        return input_ids

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


import ftfy
import html
import regex as re

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class HFGPT2Tokenizer(AbstractTokenizer):
    """Designed to Integrate the pretrained OpenAI GPT2 Tokenizers from HF"""

    def __init__(self, seq_length, fast=True):
        name = "HFGPT2Tokenizer"
        if fast:
            name += "Fast"
        super().__init__(name)
        vocab_name = "gpt2"
        if fast:
            self.tokenizer = GPT2TokenizerFast.from_pretrained(vocab_name)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(vocab_name)

        self.tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
        self.eod_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.seq_length = seq_length

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        return self.tokenizer._tokenizer.decoder

    def tokenize(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        texts = [whitespace_clean(basic_clean(text)) for text in texts]
        input_ids = self.tokenizer(
            texts,
            max_length=self.seq_length,
            truncation=True,
        ).input_ids
        # add eod_id and pad with pad_id
        for idx,ids in enumerate(input_ids):
            if len(ids) < self.seq_length:
                ids = ids+[self.eod_id]+[self.pad_id]*(self.seq_length-len(ids)-1)
            else: # truncated
                ids = ids[:-1]+[self.eod_id]
            input_ids[idx]=ids
        input_ids = torch.tensor(input_ids,dtype=torch.int64) 
        return input_ids

    # def tokenize_batch(self, text_batch: Union[List[str], str]):
    #     if isinstance(text_batch, str):
    #         text_batch = [text_batch]
    #     return [self.tokenize(t) for t in text_batch]

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class CharLevelTokenizer(AbstractTokenizer):
    """Character Level Tokenizer"""

    def __init__(self, vocab_size):
        name = "CharLevelTokenizer"
        super().__init__(name)
        self._vocab_size = vocab_size
        self.eod_id = 0
        self.pad_id = 1

    def clamp(self, n):
        return max(32, min(n, self.vocab_size))

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    def decode_token(self, token: int):
        return str(chr(self.clamp(token)))

    def tokenize(self, text: str):
        return list(np.fromstring(text, dtype=np.uint8))

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        if isinstance(text_batch, list):
            return [self.tokenize(s) for s in text_batch]
        else:
            return self.tokenize(text_batch)

    def detokenize(self, token_ids):
        return "".join(list(map(self.decode_token, token_ids)))

    @property
    def eod(self):
        return self.eod_id


class TiktokenTokenizer(AbstractTokenizer):
    """Tokenizer from OpenAI's tiktoken implementation"""

    def __init__(self, vocab_file):
        try:
            import tiktoken
        except ModuleNotFoundError:
            print("Please install tiktoken: (https://github.com/openai/tiktoken)")
            raise Exception

        name = "TiktokenTokenizer"
        super().__init__(name)

        self.tokenizer = tiktoken.get_encoding(vocab_file)
        self.eod_id = self.tokenizer.eot_token
        self.pad_id = None

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    @property
    def vocab(self):
        raise NotImplementedError(
            "TiktokenTokenizer does not implement vocabulary access."
        )

    @property
    def inv_vocab(self):
        raise NotImplementedError(
            "TiktokenTokenizer does not implement vocabulary access. \
                To get the idx-th token in vocabulary, use tokenizer.decode([idx]) ."
        )

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)  # ,  allowed_special="all")

    def tokenize_batch(self, text_batch: List[str]):
        return self.tokenizer.encode_batch(text_batch, allowed_special="all")

    def detokenize(self, token_ids):
        return self.tokenizer.decode(tokens=token_ids, errors="strict")

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad(self):
        raise NotImplementedError
