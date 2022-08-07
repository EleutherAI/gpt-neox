# coding=utf-8
# Copyright (c) 2022, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version. 
# TODO: add attribution to Bigscience Meg-DS fork + authors
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

import os
import time

import numpy as np
import torch

from megatron import print_rank_0, mpu
from megatron.data.mtf_dataset import MTFDataset
from megatron.data.gpt2_dataset import _build_shuffle_idx

from megatron.data.temp_data_utils import get_indexed_dataset


class DecoderPackedMTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset, # TODO: remove this arg?
        num_samples,
        seq_length: int,
        seed,
        data_impl,
        skip_warmup=False,
        build_index_mappings=True,
        tokenizer=None,
    ):
        # build underlying indexed datasets
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, data_impl=data_impl, skip_warmup=skip_warmup, documents=documents)

        assert tokenizer, "Must pass a tokenizer to pack multi-task examples"
        self.tokenizer = tokenizer

        self.pad_token = tokenizer.pad
        self.eod_token = tokenizer.eod
        self.seq_length = seq_length

        self.sample_index, self.shuffle_index = _build_index_mappings(
            name=name, 
            data_prefix=data_prefix, 
            documents=len(documents), 
            mtf_dataset=self.mtf_dataset, 
            num_samples=num_samples, 
            seq_length=seq_length, 
            seed=seed,
            )

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):

        start, end = self.sample_index[idx]
        mtf_samples_indices = self.shuffle_index[start: end]
        items = [self.mtf_dataset[sample_id] for sample_id in mtf_samples_indices]

        return self.pack_samples(items)

    def pack_samples(self, items):
        """
        Greedily packs samples.
        Items:
            [
                {
                    'input_tokens': array([6, 7]),
                    'target_tokens': array([8])
                },
                {
                    'input_tokens': array([3, 4]),
                    'target_tokens': array([5])
                }
            ]
        Output:
            decoder_tokens = [[6, 7, 8, 3, 4, 5, <pad>]]: Concatenation of tokens followed with padding tokens.
            decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]: Segment ids determine original documents.
            decoder_is_inputs = [[1, 1, 0, 1, 1, 0, 0]]: `1` depicts inputs, `0` depicts target.
        """

        decoder_tokens = np.full((self.seq_length,), self.pad_token, dtype=np.int64)
        decoder_segment_ids = np.zeros((self.seq_length,), dtype=np.int64)
        decoder_is_inputs = np.full((self.seq_length,), False, dtype=bool)

        # `0` is reserved for padding
        item_num = 1
        cur_len = 0

        assert len(items) > 0

        for token_dict in items:
            input_token_len = len(token_dict["input_tokens"])
            target_token_len = len(token_dict["target_tokens"])

            total_len = input_token_len + target_token_len

            if cur_len + total_len > self.seq_length:
                # This should not happen at the indexing should only allow the correct number of items
                raise ValueError(f"""Items to be packed do not fit inside a single sample.
                    current length: {cur_len}
                    input tokens length: {input_token_len}
                    target token length: {target_token_len}
                    expected sequence length: {self.seq_length}
                """)

            decoder_tokens[cur_len: cur_len + input_token_len] = token_dict["input_tokens"]
            decoder_tokens[cur_len + input_token_len: cur_len + total_len] = token_dict["target_tokens"]
            decoder_segment_ids[cur_len: cur_len + total_len] = item_num
            decoder_is_inputs[cur_len: cur_len + input_token_len] = 1  # inputs
            # targets are already 0 at init, no need to update `decoder_is_inputs`

            item_num += 1
            cur_len += total_len
            assert cur_len <= self.seq_length

        return {
            "decoder_token_ids": decoder_tokens,
            "decoder_segment_ids": decoder_segment_ids,
            "decoder_is_inputs": decoder_is_inputs.astype('int64'),
        }


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    mtf_dataset,
    num_samples: int,
    seq_length: int,
    seed,
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += '_{}s'.format(seed)
    sample_idx_filename = _filename + '_decoder_packed_batch_idx.npy'
    shuffle_idx_filename = _filename + '_decoder_packed_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0:
        if (not os.path.isfile(sample_idx_filename)) or \
           (not os.path.isfile(shuffle_idx_filename)):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            start_time = time.time()
            row_offset = 0
            old_sample_start = 0
            epoch = 0
            shuffle_idx = []
            sample_idx = []
            while len(sample_idx) <= num_samples:
                new_document_ids = _build_shuffle_idx(size=documents, np_rng=np_rng)
                # Generate a shuffling of the entire dataset
                shuffle_idx.append(new_document_ids)
                # Packs them into a single sample
                new_samples, row_offset, old_sample_start = _build_sample_idx(
                    mtf_dataset=mtf_dataset,
                    document_ids=new_document_ids,
                    seq_length=seq_length,
                    row_offset=row_offset,
                    old_sample_start=old_sample_start,
                    epoch=epoch
                )
                sample_idx.extend(new_samples)
                epoch += 1

            shuffle_idx = np.concatenate(shuffle_idx, axis=0)
            sample_idx = np.stack(sample_idx, axis=0)

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipe_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_model_parallel_group()))

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading doc-idx mapping from {}'.format(
        sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))

    return sample_idx, shuffle_idx

def _build_sample_idx(mtf_dataset, document_ids, seq_length, row_offset, old_sample_start, epoch):
    """Build start and off index of each `full` batch, return that list of batch + start of the unfinished batch"""
    row_length = row_offset

    full_samples = []
    current_sample_start = old_sample_start
    epoch_offset = epoch * len(document_ids)

    assert epoch_offset >= current_sample_start
    for current_sample_end, document_id in enumerate(document_ids):
        current_sample_end = epoch_offset + current_sample_end
        sample_sizes = mtf_dataset.size(document_id)

        tok_len = sample_sizes["input_tokens"] + sample_sizes["target_tokens"]

        row_length = row_length + tok_len
        if row_length > seq_length:
            # current sample can't be added and requires to be added in the next one
            if current_sample_end > current_sample_start:
                full_samples.append(np.asarray([current_sample_start, current_sample_end]))
            current_sample_start = current_sample_end
            row_length = tok_len

            if tok_len > seq_length:
                # silently skips examples longer than seq_length (will never fit into one batch "row")
                # logger.warning(f"Skipping sample id={document_id}. Maximum sequence length: {seq_length}, sample length: {tok_len}")
                current_sample_start = current_sample_end + 1
                row_length = 0
                continue

    return full_samples, row_length, current_sample_start


class MTFDataset(torch.utils.data.Dataset):
    """
    A helper dataset class underlying the DecoderPackedMTFDataset class. 
    Stores input document tokens and target document tokens in 2 indexed datasets.
    """
    def __init__(
        self,
        name,
        data_prefix,
        data_impl,
        skip_warmup,
        documents,
    ):
        # Params to store
        self.name = name

        # indexed dataset
        self.input_indexed_dataset = get_indexed_dataset(data_prefix, is_input=True, data_impl=data_impl, skip_warmup=skip_warmup)
        self.target_indexed_dataset = get_indexed_dataset(data_prefix, is_input=False, data_impl=data_impl, skip_warmup=skip_warmup)

        # validity checks
        assert np.min(documents) >= 0
        assert np.max(documents) < self.input_indexed_dataset.sizes.shape[0]
        assert np.max(documents) < self.target_indexed_dataset.sizes.shape[0]
        assert self.input_indexed_dataset.sizes.shape[0] == self.target_indexed_dataset.sizes.shape[0]

    def __len__(self):
        return len(self.input_indexed_dataset)

    def __getitem__(self, idx):
        input_tokens = self.input_indexed_dataset.get(idx)
        target_tokens = self.target_indexed_dataset.get(idx)

        assert len(input_tokens) > 0
        assert len(target_tokens) > 0

        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
        }

    def size(self, idx):
        return {
            'input_tokens': self.input_indexed_dataset.size(idx),
            'target_tokens': self.target_indexed_dataset.size(idx),
        }


def get_indexed_dataset(data_prefix: str, is_input: bool, data_impl: str, skip_warmup: bool):
    """retrieve either an input or target indexed dataset."""
    if is_input:
        field = "inputs"
    else:
        field = "targets"

    return get_indexed_dataset_(f"{data_prefix}_{field}_document", data_impl, skip_warmup)

def get_indexed_dataset_(path, data_impl, skip_warmup):
    """build the indexed dataset (if does not exist)."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(path,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset