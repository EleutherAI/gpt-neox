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

    encoder_tokens = sample[:encoder_seq_length]
    decoder_tokens = sample[encoder_seq_length:]

    max_predictions_per_seq = masked_lm_prob * encoder_seq_length
    vocab_dict = tokenizer.vocab
    vocab_id_list = list(vocab_dict.values())
    vocab_id_to_token_dict = {v: k for k, v in vocab_dict.items()}

    cls_id = tokenizer.sentinels[0]
    sep_id = tokenizer.sentinels[0]
    mask_id = tokenizer.sentinels[0]
    (encoder_tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
                                                            encoder_tokens,
                                                            vocab_id_list, vocab_id_to_token_dict,
                                                            masked_lm_prob,
                                                            cls_id, sep_id, mask_id,
                                                            max_predictions_per_seq,
                                                            np_rng,
                                                            max_ngrams=3,
                                                            do_whole_word_mask=True,
                                                            favor_longer_ngram=False,
                                                            do_permutation=False,
                                                            geometric_dist=True,
                                                            masking_style="bert"
                                                            )

    pad_id = tokenizer.pad
    encoder_input_tokens, encoder_target_tokens, padding_mask, loss_mask = pad_and_convert_to_numpy(
                                                                encoder_tokens,
                                                                masked_positions,
                                                                masked_labels,
                                                                pad_id,
                                                                encoder_seq_length
                                                            )

    return {
        'encoder_input_tokens': encoder_input_tokens,
        'encoder_target_tokens': encoder_target_tokens,
        'decoder_tokens': np.array(decoder_tokens, dtype=np.int64),
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


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False,
                                 geometric_dist=False,
                                 masking_style="bert"):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1. / np.arange(1, max_ngrams + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_longer_ngram:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(ngrams[:len(cand_index_set)],
                              p=pvals[:len(cand_index_set)] /
                              pvals[:len(cand_index_set)].sum(keepdims=True))
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)
            n = min(np_rng.geometric(0.2), max_ngrams)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(
            index=index_set,
            label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, labels_np, padding_mask_np, loss_mask_np