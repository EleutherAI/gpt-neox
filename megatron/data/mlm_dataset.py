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

"""T5-style MLM denoising dataset. Originally added for MLM-adapting decoder models."""

import os
import time
import random

import numpy as np
import torch

from megatron import mpu, print_rank_0
from megatron.data.gpt2_dataset import GPT2Dataset


class MLMDataset(torch.utils.data.Dataset):

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
        tokenizer=None,
        padded_vocab_size=None,
        noise_density=0.15,
        mean_noise_span_length=3,
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.seq_length = seq_length
        self.indexed_dataset = indexed_dataset

        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # T5-like span masked language modeling will fuse consecutively masked tokens to a single sentinel token.
        # To ensure that the input length is `seq_length`, we need to increase the maximum length
        # according to `noise_density` and `mean_noise_span_length`. We can also define the label length accordingly.
        expanded_inputs_length, inputs_length, targets_length, num_noise_spans = compute_input_and_target_lengths(
            seq_length=self.seq_length,
            noise_density=self.noise_density,
            mean_noise_span_length=self.mean_noise_span_length,
            )

        self.inputs_length = inputs_length
        # TODO(Hailey): is this still the case in this codebase vs Meg-DS? think so but should check
        # in order to compute loss, need an extra token at the end
        self.expanded_inputs_length = expanded_inputs_length + 1
        self.targets_length = targets_length + 1

        self.num_noise_spans = num_noise_spans
        
        # build the samples mapping.
        self._gpt_dataset = GPT2Dataset(
            name=self.name,
            data_prefix=data_prefix,
            documents=documents,
            indexed_dataset=self.indexed_dataset,
            num_samples=num_samples,
            # TODO(Hailey:) why didn't it work with -1?
            # GPT2Dataset will return length `seq_length + 1` sequences, so -1
            seq_length=self.expanded_inputs_length,
            seed=seed,
            build_index_mappings=build_index_mappings,
        )

        # Vocab stuff.
        self.tokenizer = tokenizer
        self.padded_vocab_size = padded_vocab_size
        self.eos_id = tokenizer.eod_id

        self.sentinel_token_ids = list(range(
            self.tokenizer.vocab_size - 1, self.padded_vocab_size
        ))

        # check that sentinel tokens are sufficient.
        assert self.eos_id is not None, "MLM dataset requires the tokenizer to have an <EOS> token"
        assert len(self.sentinel_token_ids) > 0, "Span denoising requires extra sentinel tokens, but none in vocab"
        assert len(self.sentinel_token_ids) >= self.num_noise_spans, f"Need at least {self.num_noise_spans} sentinel tokens, "\
                                "but only {len(self.sentinel_token_ids)} extra tokens available. Add more with --extra-sentinel-tokens."

    def __len__(self):
        return len(self._gpt_dataset)

    def __getitem__(self, idx):

        #TODO(Hailey): does this report IndexError in a way that's legible to user?
        sample = self._gpt_dataset[idx]["text"]

        return build_training_sample(
                        sample,
                        inputs_length=self.inputs_length,
                        targets_length=self.targets_length,
                        num_noise_spans=self.num_noise_spans,
                        sep_id=self.eos_id,
                        all_sentinel_token_ids=self.sentinel_token_ids,
                        )     


def build_training_sample(
    sample,
    inputs_length,
    targets_length,
    num_noise_spans,
    sep_id,
    all_sentinel_token_ids,
):
    """Build training sample.
    Arguments:
        sample: int32 tensor
        inputs_length: integer
        targets_length: integer
        num_noise_spans: integer
        sep_id: integer
        all_sentinel_token_ids: List[int]
    Returns:
        Dict with following keys:
            - `input_tokens`: int32 tensor with as length input_length,
            - `target_tokens`: int32 tensor with as length targets_length + 1,
    """

    spans_start, mask_indices = random_spans_noise_mask(
        inputs_length=inputs_length,
        targets_length=targets_length,
        num_noise_spans=num_noise_spans,
    )
    spans_end = np.concatenate([
        spans_start[1:], np.full((1,), len(sample), dtype=np.int32)]
    )

    sentinel_token_ids = all_sentinel_token_ids[:num_noise_spans]

    input_token_ids = np.concatenate(
        [
            elt
            for start, end, sentinel_token in zip(spans_start[::2], spans_end[::2], sentinel_token_ids)
            for elt in [sample[start: end], np.full((1,), sentinel_token, dtype=np.int32)]
        ] +
        [np.full((1,), sep_id, dtype=np.int32)]
    )
    target_token_ids = np.concatenate(
        [
            elt
            for start, end, sentinel_token in zip(spans_start[1::2], spans_end[1::2], sentinel_token_ids)
            for elt in [np.full((1,), sentinel_token, dtype=np.int32), sample[start: end]]
        ] +
        [np.full((1,), sep_id, dtype=np.int32)]
    )
    
    return {
        'input_tokens': input_token_ids,
        'target_tokens': target_token_ids
    }


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


def random_spans_noise_mask(
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
    num_noise_tokens = targets_length - num_noise_spans - 1
    num_nonnoise_tokens = inputs_length - num_noise_spans - 1
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

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
    )
    span_starts = np.concatenate([np.full((1,), 0, dtype=np.int32), np.cumsum(interleaved_span_lengths)[:-1]])
    span_start_indicator = np.zeros((number_of_raw_tokens,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return span_starts, is_noise
