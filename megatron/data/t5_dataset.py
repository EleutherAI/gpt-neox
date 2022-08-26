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

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron.data.gpt2_dataset import _build_index_mappings

class T5Dataset(torch.utils.data.Dataset):

    def __init__(self,
        name,
        data_prefix,
        documents, #?
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        neox_args=None,
    ):
# num_epochs, max_num_samples, masked_lm_prob,
#                  max_seq_length, max_seq_length_dec,
#                  short_seq_prob,
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

        # self.inputs_length is the `seq-length: xxx` value set in the config.
        # self.max_target_seq_length is a hard cap on how much noise / how long of a sequence ...
        # ... must be generated/reconstructed by the decoder.
        self.inputs_length = seq_length
        self.max_target_seq_length = neox_args.decoder_seq_length

        # self.raw_seq_length stores the chunk size to retrieve from dataset.
        # self.targets_length stores the length of targets, based on amount of noise.
        self.raw_seq_length, self.targets_length = compute_input_and_target_lengths(
            input_seq_length=seq_length,
            mean_noise_span_length=self.mean_noise_span_length,
            masked_lm_prob=self.masked_lm_prob,
            extra_tokens_per_span_inputs=1,
            extra_tokens_per_span_targets=1,
        )

        # + 1 is bc we add an EOD token to inputs and outputs
        assert self.max_target_seq_length >= self.targets_length + 1, \
            f'Expected targets length for span corruption ({self.targets_length + 1}) \
                is greater than configured `decoder_seq_length` ({neox_args.decoder_seq_length})'

        # TODO(Hailey): check whether these +1 s are necessary for these vars
        # we add an EOD token at end of inputs + targets.
        # self.raw_seq_length = self.raw_seq_length + 1
        # self.targets_length = self.targets_length + 1

        
        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                self.raw_seq_length,
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
        # self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        # self.vocab_id_to_token_dict = tokenizer.inv_vocab
        # self.cls_id = tokenizer.cls
        # self.sep_id = tokenizer.sep
        # self.mask_id = tokenizer.mask
        # self.pad_id = tokenizer.pad
        # self.bos_id = tokenizer.bos_token_id
        # self.eos_id = tokenizer.eos_token_id
        # self.sentinel_tokens = tokenizer.additional_special_tokens_ids

        # check sentinel token existence
        assert len(self.tokenizer.sentinels) > 0, "Run with `extra-sentinel-tokens: 100` to include enough sentinels for T5."

    def __len__(self):
        return self.samples_mapping.shape[0]

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
                print(sample_list)
                sample = np.concatenate(sample_list, dtype=np.int64)

            print(sample)
            

            return build_sample(
                sample=sample,
                seq_length=self.raw_seq_length,
                target_seq_length=self.target_seq_length,
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
    seq_length,
    target_seq_length,
    masked_lm_prob,
    mean_noise_span_length,
    tokenizer,
    np_rng,
):
    spans_start = random_spans_noise_mask(
        raw_seq_length=seq_length,
        target_seq_length=target_seq_length,
        masked_lm_prob=masked_lm_prob,
        mean_noise_span_length=mean_noise_span_length,
        np_rng=np_rng,
    )

    spans_end = np.concatenate([
        spans_start[1:], np.full((1,), len(sample), dtype=np.int32)]
    )
    assert len(sample) == seq_length, "sample is not same length as `self.raw_seq_length`"
    
    # TODO(Hailey): in order to do T5-denoising *with no sentinels*, we should refactor these listcomps ...
    # ... such that they use a helper fn add_sentinel() which abstracts the appending / prepending of sentinels.

    # loop through even spans (non-noise), adding that non-noise span + sentinel for the subsequent noise span
    # at each step
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
        'input_tokens': input_token_ids,
        'target_tokens': target_token_ids
    }




def compute_input_and_target_lengths(
    input_seq_length, 
    noise_density, 
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
        num_noise_tokens = int(round(_tokens_length * noise_density))
        num_nonnoise_tokens = _tokens_length - num_noise_tokens
        _num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans and one SEP token.
        _input_length = num_nonnoise_tokens + _num_noise_spans + extra_tokens_per_span_inputs
        _output_length = num_noise_tokens + _num_noise_spans + extra_tokens_per_span_targets
        return _input_length, _output_length, _num_noise_spans

    tokens_length = input_seq_length
    inputs_length, targets_length, num_noise_spans = _tokens_length_to_inputs_length_targets_length(tokens_length)
    while inputs_length + targets_length > input_seq_length:
        tokens_length -= 1
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

        return segment_lengths, np_rng

    noise_span_lengths, np_rng = _randomly_segment(num_noise_tokens, num_noise_spans, np_rng)
    nonnoise_span_lengths, np_rng = _randomly_segment(target_seq_length, num_noise_spans, np_rng)

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=-1), [num_noise_spans * 2]
    ) # interleave the 2 lists of span lengths


    # add left boundary of first span (idx 0) and drop right boundary of last span (index seq_length)
    span_starts = np.concatenate([np.full((1,), 0)], np.cumsum(interleaved_span_lengths)[:-1]) 
    span_start_indicator = np.zeros((raw_seq_length), dtype=bool)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator) # segment_ids
    is_noise = np.equal(span_num % 2, 1)

    return span_starts


# def build_training_sample(sample, target_seq_length,
#                           max_seq_length, max_seq_length_dec,
#                           vocab_id_list, vocab_id_to_token_dict,
#                           cls_id, sep_id, mask_id, pad_id,
#                           masked_lm_prob, np_rng, bos_id=None,
#                           eos_id=None, sentinel_tokens=None):
#     """Build training sample.
#     Arguments:
#         sample: int64 1-d tensor.
#         target_seq_length: Desired sequence length.
#         max_seq_length: Maximum length of the sequence. All values are padded to
#             this length.
#         max_seq_length_dec: Maximum length of the decoder sequence.
#         masked_lm_prob: Probability to mask tokens.
#         np_rng: Random number genenrator. Note that this rng state should be
#               numpy and not python since python randint is inclusive for
#               the opper bound whereas the numpy one is exclusive.
#         bos_id: start of decoder example id
#         eos_id: end of generation id
#         sentinel_tokens: unique value to be substituted for every replaced span
#     """
#     # TODO(Hailey): determine why target_seq_length is needed
#     assert target_seq_length <= max_seq_length

#     # Truncate to `target_sequence_length`.
#     # max_num_tokens = target_seq_length
#     # truncated = len(tokens) > max_num_tokens
#     # tokens = tokens[:max_num_tokens]

#     # Masking.
#     max_predictions_per_seq = masked_lm_prob * target_seq_length
#     (tokens, masked_positions, masked_labels, _, masked_spans) = create_masked_lm_predictions(
#         sample, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
#         cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng,
#         max_ngrams=10, geometric_dist=True, masking_style="t5")



#     # Padding.
#     tokens_enc, tokens_dec_in, labels, enc_mask, \
#     dec_mask, enc_dec_mask, loss_mask \
#         = pad_and_convert_to_numpy(tokens, masked_positions,
#                                    masked_labels, pad_id, max_seq_length,
#                                    max_seq_length_dec, masked_spans,
#                                    bos_id, eos_id, sentinel_tokens)

#     train_sample = {
#         'text_enc': tokens_enc,
#         'text_dec': tokens_dec_in,
#         'labels': labels,
#         'loss_mask': loss_mask,
#         'truncated': int(truncated),
#         'enc_mask': enc_mask,
#         'dec_mask': dec_mask,
#         'enc_dec_mask': enc_dec_mask,
#     }
#     return train_sample


# def pad_and_convert_to_numpy(tokens, masked_positions,
#                              masked_labels, pad_id,
#                              max_seq_length, max_seq_length_dec,
#                              masked_spans=None, bos_id=None,
#                              eos_id=None, sentinel_tokens=None):
#     """Pad sequences and convert them to numpy."""

#     sentinel_tokens = collections.deque(sentinel_tokens)
#     t5_input = []
#     (t5_decoder_in, t5_decoder_out) = ([bos_id], [])
#     (start_index, end_index) = (0, None)
#     for span in masked_spans:
#         flag = sentinel_tokens.popleft()

#         # Append the same tokens in decoder input and output
#         t5_decoder_in.append(flag)
#         t5_decoder_in.extend(span.label)
#         t5_decoder_out.append(flag)
#         t5_decoder_out.extend(span.label)

#         end_index = span.index[0]
#         t5_input.extend(tokens[start_index: end_index])
#         t5_input.append(flag)

#         # the next start index is the token after the last span token
#         start_index = span.index[-1] + 1

#     # Add <eos> token to the t5_decoder_out
#     t5_decoder_out.append(eos_id)

#     # Add the remaining tokens to the t5 input
#     t5_input.extend(tokens[start_index:])

#     # assert (len(t5_input) - len(masked_spans)) + \
#     #        (len(t5_decoder_in) - (len(masked_spans) + 1)) == len(tokens)

#     # Some checks.

#     # Encoder-side padding mask.
#     num_tokens = len(t5_input)
#     padding_length = max_seq_length - num_tokens
#     assert padding_length >= 0
#     assert len(masked_positions) == len(masked_labels)

#     # Tokens..
#     filler = [pad_id] * padding_length
#     tokens_enc = np.array(t5_input + filler, dtype=np.int64)

#     # Decoder-side padding mask.
#     num_tokens_dec = len(t5_decoder_in)
#     padding_length_dec = max_seq_length_dec - num_tokens_dec
#     assert padding_length_dec >= 0
#     filler_dec = [pad_id] * padding_length_dec
#     tokens_dec_in = np.array(t5_decoder_in + filler_dec, dtype=np.int64)

#     # Create attention masks
#     enc_mask = make_attention_mask(tokens_enc, tokens_enc)
#     enc_dec_mask = make_attention_mask(tokens_dec_in, tokens_enc)
#     dec_mask = make_attention_mask(tokens_dec_in, tokens_dec_in)
#     dec_mask = dec_mask * make_history_mask(tokens_dec_in)

#     # Labels mask.
#     labels = t5_decoder_out + ([-1] * padding_length_dec)
#     labels = np.array(labels, dtype=np.int64)

#     # Loss mask
#     loss_mask = ([1] * num_tokens_dec) + ([0] * padding_length_dec)
#     loss_mask = np.array(loss_mask, dtype=np.int64)

#     return tokens_enc, tokens_dec_in, labels, enc_mask, \
#            dec_mask, enc_dec_mask, loss_mask

def make_attention_mask(source_block, target_block):
    """
    Returns a 2-dimensional (2-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    mask = mask.astype(np.int64)
    # (source_length, target_length)
    return mask


def make_attention_mask_3d(source_block, target_block):
    """
    Returns a 3-dimensional (3-D) attention mask
    :param source_block: 1-D array
    :param target_block: 1-D array
    """
    mask = (target_block[:, None, :] >= 1) * (source_block[:, :, None] >= 1)
    # (batch, source_length, target_length)
    # mask = mask.astype(np.int64)
    return mask


def make_history_mask(block):
    length = block.shape[0]
    arange = np.arange(length)
    history_mask = (arange[None, ] <= arange[:, None])
    history_mask = history_mask.astype(np.int64)
    return history_mask


def make_history_mask_3d(block):
    batch, length = block.shape
    arange = torch.arange(length, device=block.device)
    history_mask = (arange[None, ] <= arange[:, None])[None, ]
    history_mask = history_mask.expand(batch, length, length)
    return history_mask
