# Copyright (c) 2024, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np
import json

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)
import time
import tqdm
import torch
import ftfy

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from threading import Semaphore


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, text):
        # Handle both raw text and JSONL with gradient signs
        gradient_sign = 1.0  # Default to gradient descent
        
        # Try to parse as JSON to extract gradient_sign
        if isinstance(text, str) and text.strip().startswith('{'):
            try:
                doc = json.loads(text)
                if isinstance(doc, dict):
                    # Extract gradient_sign if present
                    gradient_sign = doc.get('gradient_sign', 1.0)
                    # Extract text content
                    text = doc.get('text', text)
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, treat as raw text
                pass
        
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        for key in self.args.jsonl_keys:
            doc_ids = []
            text_ids = Encoder.tokenizer.tokenize(text)
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(text), gradient_sign


def get_args(input_args=None):
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
        help="space separate listed of keys to extract from jsonl. Default: text",
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
            "TiktokenTokenizer",
            "SPMTokenizer",
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
    args = parser.parse_args(input_args)
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
        # Check if it's a JSONL file
        if fname.endswith('.jsonl'):
            # For JSONL files, read line by line to preserve full JSON structure
            with open(fname, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # filter out empty lines
                        semaphore.acquire()
                        yield line
        else:
            # For other formats, use lm_dataformat
            for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
                semaphore.acquire()
                yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main(input_args=None):
    args = get_args(input_args)
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")

    # build a semaphore object to stop `yield_from_files` from getting ahead of encoder.encode and
    # hence building up memory
    semaphore = Semaphore(10000 + args.workers)

    # Auto-detect if we should save gradient signs by checking first few documents
    has_gradient_signs = False
    print("Checking if dataset contains gradient_sign fields...")
    
    # Sample first few documents to check for gradient_sign field
    sample_fin = yield_from_files(args.input.split(","), semaphore)
    sample_count = 0
    gradient_sign_found = False
    
    for doc in sample_fin:
        sample_count += 1
        # Check if doc is a JSON string with gradient_sign field
        if isinstance(doc, str) and doc.strip().startswith('{') and '"gradient_sign"' in doc:
            gradient_sign_found = True
            break
        
        # Check first 10 documents
        if sample_count >= 10:
            break
    
    if gradient_sign_found:
        has_gradient_signs = True
        print(f"Found gradient_sign field in document {sample_count} - will save gradient signs")
    else:
        print(f"No gradient_sign field found in {sample_count} sampled documents - gradient signs will not be saved")
    
    # Reset to process all documents
    fin = yield_from_files(args.input.split(","), semaphore)

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encoder.encode(doc) for doc in fin)

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
    
    # Create builder for gradient signs if auto-detected
    gradient_signs_builder = None
    if has_gradient_signs:
        gradient_signs_bin_file = f"{args.output_prefix}_gradient_signs.bin"
        gradient_signs_idx_file = f"{args.output_prefix}_gradient_signs.idx"
        gradient_signs_builder = indexed_dataset.make_builder(
            gradient_signs_bin_file,
            impl=args.dataset_impl,
        )
        # Set dtype for gradient signs
        gradient_signs_builder._dtype = np.float32
        print(f"Creating gradient signs dataset: {gradient_signs_bin_file}")

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed, gradient_sign) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, sentences in doc.items():
            for sentence in sentences:
                builders[key].add_item(np.array(sentence, dtype=builders[key].dtype))
            # separate with eos token
            builders[key].end_document()
        
        # Store gradient sign for this document if requested
        if gradient_signs_builder is not None:
            gradient_signs_builder.add_item(np.array([gradient_sign], dtype=np.float32))
            gradient_signs_builder.end_document()

        # log progress
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            pbar.set_description(
                f"Processed {i}{'' if args.num_docs is None else '/' + str(args.num_docs)} documents ({i / elapsed :.2f} docs/s, {mbs:.2f} MB/s)."
            )
            if i != 0:
                pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])
    
    # Finalize gradient signs if they were created
    if gradient_signs_builder is not None:
        gradient_signs_builder.finalize(gradient_signs_idx_file)
        print(f"Gradient signs saved to: {gradient_signs_bin_file} and {gradient_signs_idx_file}")
    else:
        if has_gradient_signs:
            print("Warning: Gradient signs were detected but not saved")
        else:
            print("No gradient signs saved (dataset does not contain gradient_sign fields)")


if __name__ == "__main__":
    main()
