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

"""MLM-LM-T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron.data.gpt2_dataset import _build_index_mappings

class MLM_LM_T5Dataset(torch.utils.data.Dataset):

    def __init__(self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        neox_args=None,
        build_index_mappings=True,
    ):

        # Params to store.
        self.name = name
        self.seed = seed
        self.neox_args = neox_args

        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.masked_lm_prob = self.neox_args.masked_lm_prob
        self.mean_noise_span_length = self.neox_args.mean_noise_span_length

        self.encoder_seq_length = seq_length
        self.decoder_seq_length = neox_args.decoder_seq_length

        # we add an EOD token at end of inputs + targets.
        self.total_seq_length = self.encoder_seq_length + self.decoder_seq_length + 1
        
        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                self.total_seq_length - 1, # indexed dataset adds 1 to this 
                seed,
            )
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

            if self.shuffle_idx_len != self.sample_idx_len:
                print(
                    f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
                )

        # Vocab stuff.
        self.tokenizer = neox_args.tokenizer

        # check sentinel token existence
        assert len(self.tokenizer.sentinels) > 0, "Run with `extra-sentinel-tokens: 100` to include enough sentinels for T5."

    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        # rng state (must be numpy). Meg-DS does this with the seed
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        # same logic as GPT2Dataset for retrieving samples from index mappings.
        # TODO(Hailey): does this function take seq_length into consideration?
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            # If we are within the same document, just extract the chunk.
            if doc_index_f == doc_index_l:
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                
                sample = np.concatenate(sample_list, dtype=np.int64)

            print("encoder_seq_length", self.encoder_seq_length)
            print("decoder_seq_length", self.decoder_seq_length)
            print("len sample", len(sample))

            return build_sample(
                sample=sample,
                encoder_seq_length=self.encoder_seq_length,
                decoder_seq_length=self.decoder_seq_length,
                masked_lm_prob=self.masked_lm_prob,
                mean_noise_span_length=self.mean_noise_span_length,
                tokenizer=self.tokenizer,
                np_rng=np_rng,
            )
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def build_sample(
    sample,
    encoder_seq_length,
    decoder_seq_length,
    masked_lm_prob,
    mean_noise_span_length,
    tokenizer,
    np_rng,
):
    #     spans_start = random_spans_noise_mask(
    #         raw_seq_length=seq_length,
    #         target_seq_length=target_seq_length,
    #         masked_lm_prob=masked_lm_prob,
    #         mean_noise_span_length=mean_noise_span_length,
    #         np_rng=np_rng,
    #     )
        
    #     assert len(tokenizer.sentinels) >= (spans_start.shape[0] / 2), f"{len(tokenizer.sentinels)} sentinel tokens available, but {spans_start.shape[0] / 2} needed. \
    # please increase `extra-sentinel-tokens` to at least {spans_start.shape[0] / 2}."

    #     spans_end = np.concatenate([
    #         spans_start[1:], np.full((1,), len(sample), dtype=np.int32)]
    #     )
    #     assert len(sample) == seq_length, f"sample length ({len(sample)}) is not same length as `self.raw_seq_length` ({seq_length})"

    print("building sample for MLM LM T5 Dataset")
    encoder_tokens = sample[:encoder_seq_length]
    decoder_tokens = sample[encoder_seq_length:]

    encoder_input_tokens = np.array(encoder_tokens, dtype=np.int64)
    encoder_target_tokens = np.array(encoder_tokens, dtype=np.int64)

    input_token_ids = np.concatenate(
        [
            item
            for start, end, sentinel in zip(spans_start[::2], spans_end[::2], tokenizer.sentinels)
            for item in [sample[start: end], np.full((1,), sentinel, dtype=np.int64)]
        ] +
        [np.full((1,), tokenizer.eod, dtype=np.int64)] # we append EOD to inputs
    )
    # likewise, loop through odd spans (noise), prepending each span's sentinel to it
    target_token_ids = np.concatenate(
        [
            item
            for start, end, sentinel_token in zip(spans_start[1::2], spans_end[1::2], tokenizer.sentinels)
            for item in [np.full((1,), sentinel_token, dtype=np.int64), sample[start: end]]
        ] +
        [np.full((1,), tokenizer.eod, dtype=np.int64)] # we append EOD to targets
    )

    return {
        'encoder_input_tokens': encoder_input_tokens,
        'encoder_target_tokens': encoder_target_tokens,
        'decoder_tokens': decoder_tokens,
    }


def compute_input_and_target_lengths(
    input_seq_length, 
    masked_lm_prob, 
    mean_noise_span_length,
    extra_tokens_per_span_inputs=1,
    extra_tokens_per_span_targets=1,):
    """This function based on `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`

    And also based on an adapted version in bigscience-workshop/Megatron-DeepSpeed.

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

    This function tells us the required number of tokens in each raw example to retrieve
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOD appended and includes that in the reported length.

    Args:
        input_seq_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
        extra_tokens_per_span_inputs: an int, defaults to 1,
        extra_tokens_per_span_targets: an int, defaults to 1,
    Returns:
        tokens_length: length of original text in tokens, to use for building indexed dataset
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(_tokens_length):
        """
        sub-helper function.
        """
        num_noise_tokens = int(round(_tokens_length * masked_lm_prob))
        num_nonnoise_tokens = _tokens_length - num_noise_tokens
        _num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans and one SEP token.
        _input_length = num_nonnoise_tokens + _num_noise_spans + extra_tokens_per_span_inputs
        _output_length = num_noise_tokens + _num_noise_spans + extra_tokens_per_span_targets
        return _input_length, _output_length, _num_noise_spans

    tokens_length = input_seq_length
    inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)
    while inputs_length <= input_seq_length:
        tokens_length += 1
        inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # tokens_length is the number of raw tokens we need to get
    # inputs_length will be the input
    # targets_length will be the target length
    # num_noise_spans is the number of spans we have to replace

    # only returning tokens_length and targets_length, for now.
    return tokens_length, targets_length

def random_spans_noise_mask(
    raw_seq_length, # length of sequence to add noise to (same as self.raw_seq_length)
    target_seq_length, # length of target sequence (from self.targets_length)
    masked_lm_prob,
    mean_noise_span_length,
    np_rng, # TODO(Hailey): figure out how best to handle rng here.
):  
    """
    Returns a size (input_seq_length,) boolean array indicating if each token is in a noise span. TODO(Hailey): update docstring w/ actual datatype of output
    """
    # TODO(Hailey): add a credit for what this function is based on (t5 repo fn of same name) and a docstring
    # TODO(Hailey): credit the bigscience Meg-DS fork

    num_noise_tokens = round(raw_seq_length * masked_lm_prob)
    num_noise_spans = round(num_noise_tokens / mean_noise_span_length)

    assert num_noise_spans >= 1, f"input seq. length ({raw_seq_length}) not long enough for a single length ({mean_noise_span_length}) span to fit"

    def _randomly_segment(length, num_segments):
        """partitions a `length` long sequence into `num_segments` distinct non-empty segments."""

        segment_indices = np.arange(length - 1) < (num_segments - 1) # a size (length - 1) array with (num_segments - 1) True values at start

        np_rng.shuffle(segment_indices)

        segment_indices = np.pad(segment_indices, [[1,0]], constant_values=0) # add False to this array. We want to start with non-noise

        segment_ids = np.cumsum(segment_indices) # turn into an array of segment ids, from 0 to num_segments - 1

        _, segment_lengths = np.unique(segment_ids, return_counts=True) # get the lengths of each segment (which sum to `length`)

        return segment_lengths

    noise_span_lengths = _randomly_segment(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _randomly_segment(raw_seq_length - num_noise_tokens, num_noise_spans)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=-1), [num_noise_spans * 2]
    ) # interleave the 2 lists of span lengths


    # add left boundary of first span (idx 0) and drop right boundary of last span (index seq_length)
    span_starts = np.concatenate([np.full((1,), 0), np.cumsum(interleaved_span_lengths)[:-1]]) 
    span_start_indicator = np.zeros((raw_seq_length), dtype=bool)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator) # segment_ids
    is_noise = np.equal(span_num % 2, 1)

    return span_starts
