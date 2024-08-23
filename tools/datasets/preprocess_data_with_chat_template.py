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

"""
A script for processing a dataset such that chat templates are utilized in the creation of the data.
These are then used to perform instruction/chat model finetunes (for example, finetuning a model on only the assistant
portions of a chatml dataset).

This follows the same output format as 'preprocess_data_with_mask.py' but using chat templates to generate the data.
This way we can support multiturn chat data in the finetuning process. instead of relying on a single turn of data.

To run this script, first edit `tools/datasets/corpora.py` such that the command to call
 `tools/datasets/preprocess_data_with_chat_template.py` is as follows:

```
cmd = f"python tools/datasets/preprocess_data_with_with_chat_template.py \
    --input {jsonl_filepath} \
    --output-prefix {parent_folder}/{self.name} \
    --tokenizer-path {hf-tokenizer} \
    --jsonl-keys {jsonl_keys} \
    --dataset-impl mmap \
    --workers {self.num_workers} "

if self.only_last:
    cmd += f"--only-last "

if self.no_mask:
    cmd += f"--no-mask "
```

Then, specify
```
"train_data_paths": ["/path/to/dataset/name_text_document"],
"label_data_paths": ["/path/to/dataset/name_label_document"]
```
in your YML config. This will then allow for finetuning on the data with loss masks set appropriately.

"""

import argparse
import multiprocessing
import os
import sys

import lm_dataformat as lmd
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
    )
)

import time
import tqdm
import jsonlines

from megatron.data import indexed_dataset
from threading import Semaphore
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer


def build_chat(
    chat: List[Dict[str, str]],
    generation_role: str,
    apply_mask: bool,
    tokenizer: PreTrainedTokenizer,
    only_last_turn: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Build a chat from a list of dictionaries. Each dictionary should have a "role" and "content" key, this follows the
    Chat Template from https://huggingface.co/docs/transformers/main/en/chat_templating

    :param chat: A list of dictionaries with "role" and "content" keys
    :param generation_role: The role of the model generating the chat, usually "assistant"
    :param apply_mask: Whether to apply a loss mask to the chat, if False, all tokens will be included in the loss
    :param tokenizer: A HF tokenizer
    :param only_last_turn: Whether to only include the last turn in the chat, needed for some fine-tuning tasks
    """
    tokens = []
    mask = []
    if apply_mask is False:
        tokens = tokenizer.apply_chat_template(chat)
        mask = tokens
        return tokens, mask
    for i, turn in enumerate(chat):
        add_gen = (
            False if i == len(chat) - 1 else chat[i + 1]["role"] == generation_role
        )
        chat_tokens = tokenizer.apply_chat_template(
            chat[: i + 1], add_generation_prompt=add_gen
        )[len(tokens) :]
        # remove previous stuff...
        tokens.extend(chat_tokens)
        if only_last_turn and (i != len(chat) - 1):
            mask.extend([-100] * len(chat_tokens))
        elif apply_mask and (turn["role"] != generation_role):
            mask.extend([-100] * len(chat_tokens))
        else:
            mask.extend(chat_tokens)
    if tokenizer.eos_token_id is not None:
        mask.append(tokenizer.eos_token_id if mask[-1] != -100 else -100)
        tokens.append(tokenizer.eos_token_id)
    return tokens, mask


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)

    def encode(self, text):
        ids = {}
        for key in self.args.jsonl_keys:
            text_ids, label_ids = build_chat(
                text[key],
                self.args.generation_role,
                not self.args.no_mask,
                Encoder.tokenizer,
                self.args.only_last,
            )
            ids[key] = (text_ids, label_ids)
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
        default=["conversation"],
        help="space separate listed of keys to extract from jsonl. Default: text",
    )
    group.add_argument(
        "--no-mask",
        help="If set, this will not mask any tokens in the input data.",
        action="store_true",
    )
    group.add_argument(
        "--generation-role",
        type=str,
        default="assistant",
        help="The role of the model generating the chat, usually 'assistant'. Default: assistant",
    )
    group.add_argument(
        "--only-last",
        help="If set, this will mask everything except the last turn in the chat.",
        action="store_true",
    )
    group.add_argument(
        "--num-docs",
        default=None,
        help="Optional: Number of documents in the input data (if known) for an accurate progress bar.",
        type=int,
    )
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to HF Tokenizer.",
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
        with open(fname, encoding="utf-8") as f:
            reader = jsonlines.Reader(f)
            for f in reader:
                semaphore.acquire()
                yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)


def main():
    args = get_args()
    encoder = Encoder(args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
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
        builders[key]._dtype = np.int32
        if not args.no_mask:
            assert (
                key + "_label" not in args.jsonl_keys
            ), "label should not be included as it will be generated according to the mask."
            key += "_label"
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
            builders[key]._dtype = np.int32

    # actually do tokenization
    proc_start = time.time()
    total_bytes_processed = 0
    pbar = tqdm.tqdm()
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed

        # release semaphore so `yield_from_files` can add another file to the buffer
        semaphore.release()

        # add each tokenized document / sentence
        for key, conv in doc.items():
            tokens = conv[0]
            token_mask = conv[1]
            builders[key].add_item(np.array(tokens, dtype=builders[key].dtype))
            builders[key + "_label"].add_item(
                np.array(token_mask, dtype=builders[key + "_label"].dtype)
            )
            # add indx...
            builders[key].end_document()
            builders[key + "_label"].end_document()
            if i == 1:
                print("key: ", key)
                print("tokens: ", tokens)
                print("token_mask: ", token_mask)
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
    update_keys = args.jsonl_keys
    for key in update_keys:
        builders[key].finalize(output_idx_files[key])
        builders[key + "_label"].finalize(output_idx_files[key + "_label"])


if __name__ == "__main__":
    main()
