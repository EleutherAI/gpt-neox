# Copyright (c) 2022, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version. 
# TODO(Hailey): add attribution to Bigscience Meg-DS fork + authors?
# coding=utf-8
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

"""Next Token prediction with Puase Tokens dataset."""

import os
import time
import random
import numpy as np
import torch

from dataclasses import dataclass
from inspect import signature
from megatron import mpu, print_rank_0

class PTokDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        use_shared_fs=True,
        label_dataset=None,
        pause_id=None,
        pause_token=None,
        with_ready=False,
        ready_id=None,
        ready_token=None,
        tokenizer_cfg=None,
        span_density=0.10,
        mean_span_length=10,
    ):

        self.name = name
        self.indexed_dataset = indexed_dataset
        self.label_dataset = label_dataset

        self.with_ready = with_ready
        if pause_id is None:
            # TODO make a way for tokenizers to be accesible
            assert pause_token is not None
            from megatron.tokenizer import build_tokenizer
            tokenizer = build_tokenizer(
                ArgsContainer.from_kwargs(**tokenizer_cfg)
                )
            self.pause_id = tokenizer.tokenize(pause_token)[0]
            if self.with_ready:
                assert ready_token is not None
                self.ready_id = tokenizer.tokenize(ready_token)[0]
        else:
            self.pause_id = pause_id
            if self.with_ready:
                self.ready_id = ready_id

        self.span_density = span_density
        self.mean_span_length = mean_span_length
        self.seq_length = seq_length

        tokens_length, self.inputs_length, self.rpt_length, self.num_token_spans = compute_input_and_target_lengths(
            self.seq_length,
            self.span_density,
            self.mean_span_length
            )

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                self.inputs_length, #seq_length,
                seed,
                use_shared_fs=use_shared_fs,
            )
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

            if self.shuffle_idx_len != self.sample_idx_len - 1:
                print(
                    f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
                )

    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]

            samples = []
            # If we are within the same document, just extract the chunk.
            for n, dataset in enumerate([self.indexed_dataset]):
                if doc_index_f == doc_index_l:
                    samples.append(
                        dataset.get(
                            self.doc_idx[doc_index_f],
                            offset=offset_f,
                            length=offset_l - offset_f + 1,
                        )
                    )
                else:
                    # Otherwise, get the rest of the initial document.
                    sample_list = [
                        dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                    ]
                    # Loop over all in between documents and add the entire document.
                    for i in range(doc_index_f + 1, doc_index_l):
                        sample_list.append(dataset.get(self.doc_idx[i]))
                    # And finally add the relevant portion of last document.
                    sample_list.append(
                        dataset.get(self.doc_idx[doc_index_l], length=offset_l + 1)
                    )
                    samples.append(np.concatenate(sample_list))

            text = samples[0]
            text_length = len(text)
            # Add pause and ready tokens here
            # text = ...
            span_idx, is_rpt = random_spans_ready_pause_tokens(self.inputs_length, self.rpt_length, self.num_token_spans)
            span_idx = np.concatenate([span_idx, [self.seq_length]])
            for idx, (start_idx, end_idx) in enumerate(zip(span_idx[:-1], span_idx[1:])):
                if idx % 2 != 0:
                    if self.with_ready:
                        token_span = [self.pause_id]*(end_idx - start_idx - 1)+[self.ready_id]
                    else:
                        token_span = [self.pause_id]*(end_idx - start_idx)
                    text = np.concatenate([text[:start_idx], token_span, text[start_idx:]])

            return {"text": np.array(text, dtype=np.int64)}
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


@dataclass
class ArgsContainer:
    tokenizer_type: str
    vocab_file: str

    @classmethod
    def from_kwargs(cls, **kwargs):
        cls_fields = {field for field in signature(cls).parameters}
        native_args, new_args = {}, {}
        for name, val in kwargs.items():
            if name in cls_fields:
                native_args[name] = val
            else:
                new_args[name] = val

        ret = cls(**native_args)
        for new_name, new_val in new_args.items():
            setattr(ret, new_name, new_val)
        return ret


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    use_shared_fs=True,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)

    # Get 

    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    if not use_shared_fs:
        should_process_dataset = int(os.environ["LOCAL_RANK"]) == 0
    else:
        should_process_dataset = torch.distributed.get_rank() == 0

    # Build the indexed mapping if not exist.
    if should_process_dataset:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            print_rank_0(
                " > WARNING: could not find index map files, building "
                "the indices on rank 0 ..."
            )
            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            from megatron.data import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length
            if 2 * (num_samples + 1) < np.iinfo(np.int32).max:
                sample_idx = helpers.build_sample_idx_int32(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            else:
                sample_idx = helpers.build_sample_idx_int64(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_io_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_io_parallel_group()
    )

    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(sample_idx.shape[0]))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int64)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Beginning offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise, start from the beginning of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def compute_input_and_target_lengths(seq_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have SEP appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(_tokens_length):
        num_noise_tokens = int(round(_tokens_length * noise_density))
        num_nonnoise_tokens = _tokens_length - num_noise_tokens
        _num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans and one SEP token.
        _input_length = num_nonnoise_tokens + _num_noise_spans + 1
        _output_length = num_noise_tokens + _num_noise_spans + 1
        return _input_length, _output_length, _num_noise_spans

    tokens_length = seq_length
    inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)
    while inputs_length + targets_length > seq_length:
        tokens_length -= 1
        inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # tokens_length is the number of raw tokens we need to get
    # inputs_length will be the input
    # targets_length will be the target
    # num_noise_spans is the number of spans we have to replace
    return tokens_length, inputs_length, targets_length, num_noise_spans


def random_spans_ready_pause_tokens(
    inputs_length,
    targets_length,
    num_noise_spans,
):

    """This function is inspired from `random_spans_noise_mask <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    Spans alternate between non-noise and noise, beginning with non-noise.
    Args:
        inputs_length: int32 scalar
        targets_length: int32 scalar
        num_noise_spans: int32 scalar
    Returns:
        a int8 tensor with shape [num_noise_spans]
        a boolean tensor with shape [length]
    """
    # # pick the lengths of the noise spans and the non-noise spans
    num_noise_tokens = targets_length # - num_noise_spans - 1
    num_nonnoise_tokens = inputs_length # - num_noise_spans - 1
    number_of_raw_tokens = num_noise_tokens + num_nonnoise_tokens

    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        # TODO @thomasw21 handle random state correctly, ie synchronized across TP.
        #   we might not care as get_batch_pipe broadcasts data to all devices.
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]], constant_values=0)
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = np.concatenate([_random_segmentation(num_noise_tokens, num_noise_spans), [0]])
    nonnoise_span_lengths = [1]
    while 1 in nonnoise_span_lengths:
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans+1)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2 + 2]
    )
    interleaved_span_lengths = interleaved_span_lengths[:-1]
    span_starts = np.concatenate([np.full((1,), 0, dtype=np.int32), np.cumsum(interleaved_span_lengths)[:-1]])
    span_start_indicator = np.zeros((number_of_raw_tokens,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = ~np.equal(span_num % 2, 1)

    return span_starts, is_noise
