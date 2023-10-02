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

"""
A script for processing a dataset such that corresponding labels are also produced. These are then used to perform masked finetuning
(for example, finetuning a model to only output the text following some delimiter in the finetuning dataset such as "Answer: "
rather than generating the entire "Question: ... Answer: " turns of conversation.

To run this script, first edit `tools/datasets/corpora.py` such that the command to call `tools/datasets/preprocess_data.py` is as follows:

```
cmd = f"python tools/datasets/preprocess_data_with_mask.py \
    --input {jsonl_filepath} \
    --output-prefix {parent_folder}/{self.name} \
    --vocab {self.vocab_file} \
    --dataset-impl mmap \
    --tokenizer-type {self.tokenizer_type} \
    --merge-file {self.merge_file} \
    --append-eod \
    --mask-before-token X,Y,Z \
    --workers {self.num_workers} "

if self.num_docs is not None:
    cmd += f"--num-docs {self.num_docs} "

if self.ftfy:
    cmd += f"--ftfy "
```
where --mask-before-token must be the (comma-separated) list of tokens produced by encoding your delimiter string.
Up to and including the first occurrence of this token sequence in a document, all tokens will have their loss mask zeroed out when the label dataset is provided to NeoX.

Then, specify
```
"train_data_paths": ["/path/to/dataset/name_text_document"],
"label_data_paths": ["/path/to/dataset/name_label_document"]
```
in your YML config. This will then allow for finetuning on the data with loss masks set appropriately.
(However, be warned that NeoX packs documents to fill context windows, which may degrade performance in some finetuning situations where instead padding out to the context length may be preferred.)
"""

import argparse
import multiprocessing
import os
import sys
import re

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
import time
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore
from functools import lru_cache


@lru_cache(maxsize=None)
def build_nxt(pattern: tuple) -> tuple:
    # The function is being cached. Use tuple to avoid the cache being tampered out of scope.
    nxt = [0]
    current = 1
    match_idx = 0

    while current < len(pattern):
        if pattern[match_idx] == pattern[current]:
            current += 1
            match_idx += 1
            nxt.append(match_idx)
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            nxt.append(0)
            current += 1

    return tuple(nxt)


def kmp(seq, pattern, first_appearance=False):
    """
    Search for the location of a subsequence in a list. Not sure if there is a python built-in
    implementation of kmp somewhere...
    """
    nxt = build_nxt(tuple(pattern))
    current = 0
    match_idx = 0

    matched = []

    while current < len(seq):
        if seq[current] == pattern[match_idx]:
            current += 1
            match_idx += 1
        elif match_idx != 0:
            match_idx = nxt[match_idx - 1]
        else:
            current += 1

        if match_idx == len(pattern):
            matched.append(current - len(pattern))
            if first_appearance:
                return matched
            match_idx = nxt[match_idx - 1]

    return matched


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        if isinstance(text, str):
            text = {"text": text}
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text["text"])
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    group.add_argument(
        "--jsonl-keys",
        nargs="+",
        default=["text"],
        help="space separate listed of keys to extract from jsonl. Defa",
    )
    group.add_argument(
        "--mask-before-token",
        default=None,
        help="apply loss masks before certain token(s). If multi-token pattern, separate by commas without space, e.g. --mask-before-token 0,1,1270 to use the token pattern [0,1,1270].",
        type=str,
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )
    group.add_argument(
        "--append-eod",
        action="store_true",
        help="Append an <eod> token to the end of a document.",
    )
    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )
    group.add_argument(
        "--dataset-impl",
        type=str,
        default="mmap",
        choices=["lazy", "cached", "mmap"],
        help="Dataset implementation to use. Default: mmap",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval between progress updates",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def mask(sentence: list, pivot_tokens: list, include_pivot=True):
    inds = kmp(sentence, pivot_tokens)
    if not inds:
        return sentence
    index = inds[0]
    if include_pivot:
        index += len(pivot_tokens)

    return [-100] * index + sentence[index:]


def main():
    args = get_args()
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents
    fin = yield_from_files(args.input.split(","), semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

    if args.mask_before_token is not None:
        token_mask = [
            int(re.sub(r"[^0-9]", "", r))
            for r in args.mask_before_token.split(",")
            if re.sub(r"[^0-9]", "", r)
        ]
    else:
        token_mask = []

    # make a dataset builder for each key in args.jsonl_keys
    # each key will output to a different file beginning with args.output_prefix
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )
    if token_mask:
        assert (
            "label" not in args.jsonl_keys
        ), "label should not be included as it will be generated according to the mask."
        key = "label"
        output_bin_files[key] = "{}_{}_{}.bin".format(
            args.output_prefix, key, "document"
        )
        output_idx_files[key] = "{}_{}_{}.idx".format(
            args.output_prefix, key, "document"
        )
        builders[key] = indexed_dataset.make_builder(
            output_bin_files[key],
            impl=args.dataset_impl,
            vocab_size=tokenizer.vocab_size,
        )
    int32_labels = ["text", "label"]
    for l in int32_labels:
        builders[l]._dtype = np.int32

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
                if token_mask:
                    masked_sentence = mask(sentence, token_mask)
                    builders["label"].add_item(
                        np.array(masked_sentence, dtype=builders["text"].dtype)
                    )
            # separate with eos token
            builders[key].end_document()
            if token_mask:
                builders["label"].end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed} docs/s, {mbs} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    update_keys = args.jsonl_keys + ["label"] if token_mask else args.jsonl_keys
    for key in update_keys:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    main()
