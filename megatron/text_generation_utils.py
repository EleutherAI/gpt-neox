# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Utilities for generating text."""

import copy
import json
import math
import os
import time
from contextlib import ExitStack
from functools import partial
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0
from megatron.data.indexed_dataset import make_builder, make_dataset
from megatron.mpu.mappings import gather_from_model_parallel_region
from megatron.neox_arguments.sampling import (
    cap_top_k,
    sampling_options_from_config,
    validate_generation_options,
    validate_sampling_options,
)


def get_batch(neox_args, context_tokens: torch.Tensor):
    """
    Generate a batch, attention mask, and position IDs on the context tokens' device.

    neox_args: NeoXArgs.
    context_tokens: torch tensor with dimensions [batch, context_size]

    returns: tuple of torch tensors (tokens, attention_mask, position_ids)
    """

    tokens = context_tokens.contiguous()
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=neox_args.tokenizer.eod,
        eod_mask_loss=neox_args.eod_mask_loss,
        sliding_window_width=neox_args.sliding_window_width,
    )
    return tokens, attention_mask, position_ids


def pad_batch(
    context_tokens: List[List[int]], pad_id: int, pad_len: int, truncate: bool = False
):
    """
    pads context lengths in context_tokens with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.

    context_tokens: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, context length to be padded; all batch items will be padded to the same length
    truncate: bool, if True, truncate context tokens to pad_len if they are longer than pad_len

    returns: tuple of padded context tokens and a list of unpadded token count
    """

    context_lengths = []
    for i, tokens in enumerate(context_tokens):
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
        elif context_length > pad_len:
            if not truncate:
                raise ValueError("context_length is bigger than to be padded length")
            context_tokens[i] = tokens[:pad_len]
            context_length = pad_len
        context_lengths.append(context_length)
    return context_tokens, context_lengths


def filter_logits(
    logits,
    top_k=0,
    top_p=1.0,
    filter_value=-float("Inf"),
    min_p=0.0,
    top_h=0.0,
    typical_p=1.0,
    epsilon_cutoff=0.0,
    eta_cutoff=0.0,
):
    """Filter [batch, vocabulary] logits and return the tensor, modified in place.

    Excluded scores are set to filter_value (default -inf). Each filter sees
    the scores left by the preceding one. As in Hugging Face Transformers
    v5.14.0, the order is temperature, top_h, top_k, top_p, min_p, typical_p,
    epsilon_cutoff, eta_cutoff. stream_tokens applies temperature before
    calling the filters; this standalone function does not apply it. See
    https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/generation/utils.py#L1273

    top_h: In (0, 1], keep a high-probability prefix of at most 100 tokens
        within the entropy budget. Zero disables; one does not.
    top_k: Keep the k highest scores, including ties; zero disables. Values
        above the supplied vocabulary width are capped with a warning.
    top_p: Keep the shortest high-probability prefix reaching this mass.
        Both one and legacy zero disable nucleus filtering.
    min_p: Keep probabilities at least min_p times the highest probability.
        Valid range: [0, 1]; zero disables.
    typical_p: Keep tokens closest to the expected surprisal up to the mass
        threshold, including ties. Valid range: (0, 1]; one disables.
    epsilon_cutoff: Keep probabilities at or above this cutoff.
    eta_cutoff: For entropy H, use min(eta_cutoff, sqrt(eta_cutoff) * exp(-H))
        as the inclusive probability cutoff. Both cutoffs accept [0, 1);
        zero disables them, and a highest-probability token is always kept.

    The original top-k/top-p filtering code was adapted from Hugging Face:
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    """

    filters = dict(
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    validate_sampling_options(filters)
    filters["top_k"] = cap_top_k(top_k, logits.size(-1))
    return _apply_logits_filters(logits, filter_value=filter_value, **filters)


def _apply_logits_filters(
    logits,
    *,
    top_k,
    top_p,
    min_p,
    top_h,
    typical_p,
    epsilon_cutoff,
    eta_cutoff,
    filter_value=-float("Inf"),
):
    """Apply already-validated filters; called once per decode step."""

    if top_h > 0.0:
        batch_size, vocab_size = logits.shape
        keep_mask = torch.zeros(
            (batch_size, vocab_size), dtype=torch.bool, device=logits.device
        )

        # Match Hugging Face's TopHLogitsWarper, including its top-100 cap.
        top_n = min(100, vocab_size)
        top_logits, top_indices = torch.topk(
            logits, top_n, dim=-1, largest=True, sorted=True
        )
        distribution = torch.distributions.Categorical(logits=top_logits)
        probabilities = distribution.probs
        log_probabilities = torch.log(probabilities)
        entropy_threshold = (distribution.entropy() * top_h).unsqueeze(-1)
        entropy_terms = -probabilities * log_probabilities
        cumulative_entropy = torch.cumsum(entropy_terms, dim=-1)

        selected_tokens = cumulative_entropy <= entropy_threshold
        selected_tokens[:, 0] = True
        keep_mask.scatter_(dim=1, index=top_indices, src=selected_tokens)
        logits[~keep_mask] = filter_value

    if 0 < top_k < logits.size(-1):
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if 0.0 < top_p < 1.0:
        # HF removes the least likely tail, including a tail of mass exactly 1 - p.
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) <= 1.0 - top_p
        remove[..., -1] = False  # Keep at least one token even at rounding limits.
        logits.masked_fill_(remove.scatter(-1, sorted_indices, remove), filter_value)

    if min_p > 0.0:
        probabilities = F.softmax(logits, dim=-1)
        top_probabilities = probabilities.amax(dim=-1, keepdim=True)
        scaled_min_p = min_p * top_probabilities
        indices_to_remove = probabilities < scaled_min_p

        # Match Hugging Face's MinPLogitsWarper guarantee that at least one
        # token remains eligible for sampling.
        top_token_indices = torch.topk(probabilities, 1, dim=-1).indices
        indices_to_remove.scatter_(-1, top_token_indices, False)
        logits[indices_to_remove] = filter_value

    if typical_p < 1.0:
        # Match Hugging Face's TypicalLogitsWarper with min_tokens_to_keep=1.
        normalized = F.log_softmax(logits, dim=-1)
        probabilities = torch.exp(normalized)
        entropy = -(normalized * probabilities).nansum(-1, keepdim=True)

        shifted_scores = torch.abs((-normalized) - entropy)
        sorted_scores, sorted_indices = torch.sort(
            shifted_scores, descending=False, dim=-1
        )
        sorted_logits = logits.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        last_index = (cumulative_probs < typical_p).sum(dim=1)
        last_index.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
            1, last_index.view(-1, 1)
        )
        sorted_indices_to_remove[..., :1] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    if epsilon_cutoff > 0.0:
        # Match Hugging Face's EpsilonLogitsWarper with min_tokens_to_keep=1.
        probabilities = F.softmax(logits, dim=-1)
        indices_to_remove = probabilities < epsilon_cutoff
        top_k_to_keep = min(1, logits.size(-1))
        indices_to_remove = indices_to_remove & (
            logits < torch.topk(logits, top_k_to_keep)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value

    if eta_cutoff > 0.0:
        # Match Hugging Face's EtaLogitsWarper with min_tokens_to_keep=1.
        probabilities = F.softmax(logits, dim=-1)
        entropy = torch.distributions.Categorical(logits=logits).entropy()
        epsilon = torch.tensor(
            eta_cutoff, dtype=probabilities.dtype, device=logits.device
        )
        eta = torch.minimum(
            epsilon, torch.sqrt(epsilon) * torch.exp(-entropy)
        ).unsqueeze(-1)
        indices_to_remove = probabilities < eta
        top_k_to_keep = min(1, logits.size(-1))
        indices_to_remove = indices_to_remove & (
            logits < torch.topk(logits, top_k_to_keep)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value

    return logits


def _mask_eos_before_minimum_tokens(
    logits, eos_token_id, generated_token_counts, minimum_tokens
):
    """Mask EOS for batch items that have generated fewer than ``minimum_tokens``."""
    if minimum_tokens > 0:
        logits[generated_token_counts < minimum_tokens, eos_token_id] = -float("Inf")
    return logits


def _gather_generation_logits(logits, padded_vocab_size, tp_size):
    """Return full-vocabulary scores for one decode position per batch item."""
    width = logits.size(-1)
    if width == padded_vocab_size:
        return logits  # The output head already gathered its shards.
    if width * tp_size != padded_vocab_size:
        raise ValueError(
            f"Unexpected generation logit width {width} for TP={tp_size} "
            f"and padded vocabulary size {padded_vocab_size}"
        )
    return gather_from_model_parallel_region(logits.contiguous())


def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_model(model, model_inputs, is_pipe_parallel=False) -> torch.Tensor:
    """Run a generation forward pass and return logits on output stages only.

    Pipeline engines take one already-prepared microbatch and skip the training
    loss. Non-output pipeline stages return None.
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model differently to a normal model
    if not is_pipe_parallel:
        return model.module(model_inputs)

    # The training batch function rebuilds position IDs starting at zero. Pass the
    # actual decode inputs and an empty labels tuple directly to the pipeline.
    # Do not compute cross-entropy: it can modify the returned FP32 logits in place.
    batch_fn, total_loss = model.batch_fn, model.total_loss
    monitor_enabled = model.monitor.enabled
    try:
        model.set_batch_fn(None)
        # eval_batch otherwise accumulates outputs as "loss" across differently
        # shaped forwards and tries to log a scalar loss even on non-output stages.
        model.total_loss = None
        model.monitor.enabled = False
        model.reset_activation_shape()
        _, logits = model.eval_batch(
            iter([(model_inputs, ())]),
            return_logits=True,
            compute_loss=False,
            reduce_output=None,
            num_micro_batches=1,
        )
        return logits
    finally:
        model.set_batch_fn(batch_fn)
        model.total_loss = total_loss
        model.monitor.enabled = monitor_enabled


def broadcast_terminate_signal(terminate_runs: int, *, device="cuda", across_job=False):
    """Send signal to all workers to terminate if we've finished the process"""
    terminate_runs_tensor = torch.tensor(
        [terminate_runs], dtype=torch.long, device=device
    )
    torch.distributed.broadcast(
        terminate_runs_tensor,
        0 if across_job else mpu.get_model_parallel_src_rank(),
        group=None if across_job else mpu.get_model_parallel_group(),
    )
    return terminate_runs_tensor[0].item()


def _generated_stop_mask(stop_tokens, tokens, end, generation_start):
    """Match suffixes in tokens[..., :end] that start at or after each prompt's end."""
    matched = tokens.new_zeros(tokens.shape[:-1], dtype=torch.bool)
    for group in stop_tokens or ():
        start = end - group.numel()
        if 0 <= start < end:
            matched |= (start >= generation_start) & (
                tokens[..., start:end] == group
            ).all(-1)
    return matched


def stop_tokens_in_completion(
    stop_tokens, context_tokens, batch_index, current_index, *, generation_start=0
):
    """Check one row through current_index (inclusive); optionally exclude its prompt."""
    return _generated_stop_mask(
        stop_tokens, context_tokens[batch_index], current_index + 1, generation_start
    ).item()


def _scale_sampling_logits(logits, temperature):
    """Return FP32 scores, shifting rows only at the FP32 scaling limit."""
    fp32 = torch.finfo(torch.float32)
    if temperature == 1.0 and logits.dtype != torch.float64:
        return logits.float()
    # A valid Python float can round to zero (or infinity) as an FP32 divisor.
    wide = logits.dtype == torch.float64 or not fp32.tiny <= temperature <= fp32.max
    scores = logits.to(torch.float64 if wide else torch.float32)
    limit = fp32.max * temperature
    if torch.finfo(logits.dtype).max > limit:
        # If a row's maximum reaches the FP32 scaling limit, first subtract it.
        # Softmax and all sampling filters are invariant to this row-wise shift.
        peak = scores.amax(dim=-1, keepdim=True)
        scores = scores - torch.where(peak.abs() >= limit, peak, 0.0)
    if temperature < torch.finfo(torch.float64).tiny:
        # Even the reciprocal of the divisor can overflow FP64. Two normal
        # divisors keep a tied maximum at zero rather than producing 0 * inf.
        root = math.sqrt(temperature)
        return ((scores / root) / root).float()
    return (scores / temperature).float()


def stream_tokens(
    neox_args,
    model,
    context_tokens: List[List[int]],
    eos_token_id: int = None,
    maximum_tokens: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_tokens=None,
    minimum_tokens: int = 0,
    min_p: float = 0.0,
    top_h: float = 0.0,
    typical_p: float = 1.0,
    epsilon_cutoff: float = 0.0,
    eta_cutoff: float = 0.0,
):
    """Yield one decode step at a time for a batch of unpadded token-ID lists.

    Supply at least one prompt with at least one token in each row. The engine
    must be in inference mode with use_cache=not recompute. Exhaust or close
    the iterator before reusing it on the same engine. All participating ranks
    must enter the call; for pipeline generation, global rank zero supplies
    the batch and all ranks receive the same selected IDs and requested scores.

    maximum_tokens limits forward steps from the shortest prompt, so longer
    rows can receive fewer tokens. None uses the space left by the longest
    prompt. minimum_tokens counts new tokens separately for each row and masks
    EOS (default: tokenizer.eod), not explicit stop sequences. stop_tokens may
    be one list of IDs or a list of such lists; a match must be wholly generated.
    Temperature zero is greedy; a positive temperature samples. See filter_logits
    for filter order, valid ranges and neutral values.

    Each yield is (tokens, start, end, scores, done). tokens is the padded
    prompt-and-completion buffer; start and end are per-row inclusive indices
    for the returned completion. end is -1 until a nonterminal token is emitted.
    done marks EOS or a matched stop sequence, not budget exhaustion. A terminal
    step does not advance end. These tensors can be updated by later steps;
    clone them to retain a snapshot.

    If neox_args.return_logits is enabled, scores has shape [steps_so_far,
    padded_vocab_size] and describes batch row zero only, including terminal
    decisions. EOS is -inf while suppressed; sampling scores also reflect
    temperature and filtering. Otherwise scores is None.
    """

    stream = _prepare_token_stream(
        neox_args,
        model,
        eos_token_id=eos_token_id,
        recompute=recompute,
        stop_tokens=stop_tokens,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    return stream(context_tokens)


def _prepare_token_stream(
    neox_args, model, *, recompute, eos_token_id=None, stop_tokens=None, **options
):
    """Validate an independent Python or eval request once, before its first prompt."""
    validate_generation_options(options)
    return _bind_token_stream(
        neox_args,
        model,
        recompute=recompute,
        eos_token_id=eos_token_id,
        stop_tokens=stop_tokens,
        **options,
    )


def _bind_token_stream(
    neox_args,
    model,
    *,
    recompute,
    maximum_tokens,
    minimum_tokens,
    temperature,
    eos_token_id=None,
    stop_tokens=None,
    **filters,
):
    """Bind already-validated settings; only the vocabulary bound needs the model."""
    if temperature > 0 and filters["top_k"]:
        filters["top_k"] = cap_top_k(filters["top_k"], neox_args.padded_vocab_size)
    return partial(
        _stream_tokens,
        neox_args,
        model,
        eos_token_id=eos_token_id,
        recompute=recompute,
        stop_tokens=stop_tokens,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        temperature=temperature,
        filters=filters,
    )


def _stream_tokens(
    neox_args,
    model,
    context_tokens,
    *,
    eos_token_id,
    recompute,
    stop_tokens,
    maximum_tokens,
    minimum_tokens,
    temperature,
    filters,
):
    model.eval()

    if neox_args.is_pipe_parallel:
        # Synchronize the ragged batch once, before padding: a peer need not
        # know the number or lengths of prompts (e.g. interactive generation).
        request = [context_tokens, stop_tokens]
        torch.distributed.broadcast_object_list(request, src=0)
        context_tokens, stop_tokens = request

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(
        copy.deepcopy(context_tokens),
        pad_id=neox_args.tokenizer.eod,
        pad_len=neox_args.seq_length,
    )

    # convert to tensor and broadcast
    context_tokens = torch.tensor(context_tokens, dtype=torch.long, device=model.device)
    if stop_tokens:
        if not isinstance(stop_tokens[0], list):
            stop_tokens = [stop_tokens]
        stop_tokens = [
            context_tokens.new_tensor(group) for group in stop_tokens if group
        ]

    # Make sure context tokens + start tokens are the same across all ranks
    token_generation_start_index = context_tokens.new_tensor(context_lengths)
    if not neox_args.is_pipe_parallel:
        for tensor in (context_tokens, token_generation_start_index):
            torch.distributed.broadcast(
                tensor,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )

    # get attention mask / position ids
    context_tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # set variables
    if eos_token_id is None:
        eos_token_id = neox_args.tokenizer.eod
    if maximum_tokens is None:
        maximum_tokens = (
            neox_args.seq_length - token_generation_start_index.max().item()
        )
        if minimum_tokens > maximum_tokens:
            raise ValueError(
                f"minimum_tokens ({minimum_tokens}) exceeds the inferred maximum_tokens "
                f"({maximum_tokens}) available in the context window"
            )
    batch_size = context_tokens.size(0)

    decode_tp_size = mpu.get_model_parallel_world_size()
    # PipelineEngine synchronizes the entire job at each forward. Share even
    # the EOS decision across replicas so no stage leaves that collective early.
    if neox_args.is_pipe_parallel:
        sample_src = model.grid.stage_to_global(model.num_stages - 1, data=0, model=0)
        sample_group = None
        sample_here = torch.distributed.get_rank() == sample_src
    else:
        sample_src = mpu.get_model_parallel_src_rank()
        sample_group = mpu.get_model_parallel_group()
        sample_here = decode_tp_size == 1 or mpu.get_model_parallel_rank() == 0
    greedy = temperature == 0.0

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length
        - 1,  # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens - 1,
    )

    with torch.no_grad(), ExitStack() as cleanup:
        # Release this rank's cache on exhaustion, explicit close, or a forward error.
        cleanup.callback(model.module.clear_cache)
        # initialize generation variables
        state_is_done = context_tokens.new_zeros(batch_size, dtype=torch.uint8)
        token_generation_end_index = context_tokens.new_full((batch_size,), -1)
        generation_logits = (
            # Yield only the executed prefix below. A serialized tensor view
            # can also include its backing storage, so don't leave the tail uninitialized.
            torch.zeros(
                last_token_index_to_generate - first_token_index_to_generate + 1,
                neox_args.padded_vocab_size,
                device=context_tokens.device,
                dtype=torch.float32,
            )
            if neox_args.return_logits
            else None
        )

        while token_index_to_generate <= last_token_index_to_generate:
            # A full-context forward must never append to an earlier prefix.
            if recompute or token_index_to_generate == first_token_index_to_generate:
                model.module.clear_cache()
            if recompute:  # recompute all tokens
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = logits[
                        :, token_index_to_generate - 1, :
                    ]  # [bs, seq, vocab_size] -> [bs, vocab_size]
            else:  # use kv cache
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = context_tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:, :token_index_to_generate]
                else:
                    tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1
                    )
                    positions_to_use = position_ids[
                        :, token_index_to_generate - 1
                    ].view(batch_size, -1)

                model_inputs = (
                    tokens_to_use,  # input_ids
                    positions_to_use,  # position_ids
                    attention_mask[
                        ...,
                        token_index_to_generate
                        - tokens_to_use.size(1) : token_index_to_generate,
                        :token_index_to_generate,
                    ],
                )

                logits = forward_model(model, model_inputs, neox_args.is_pipe_parallel)
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    generated_token_logits = (
                        logits[:, -1].view(batch_size, -1).contiguous()
                    )  # [bs, seq, vocab_size] -> [bs, vocab_size]

            step = token_index_to_generate - first_token_index_to_generate
            if logits is not None:
                if decode_tp_size > 1:
                    generated_token_logits = _gather_generation_logits(
                        generated_token_logits,
                        neox_args.padded_vocab_size,
                        decode_tp_size,
                    )
                generated_token_counts = (
                    token_index_to_generate - token_generation_start_index
                )
                generated_token_logits = _mask_eos_before_minimum_tokens(
                    generated_token_logits,
                    eos_token_id,
                    generated_token_counts,
                    minimum_tokens,
                )

                if not greedy:
                    generated_token_logits = _scale_sampling_logits(
                        generated_token_logits, temperature
                    )
                    generated_token_logits = _apply_logits_filters(
                        generated_token_logits, **filters
                    )
                if sample_here:
                    if greedy:
                        generated_tokens = generated_token_logits.argmax(dim=-1)
                    else:
                        probabilities = F.softmax(generated_token_logits, dim=-1)
                        generated_tokens = torch.multinomial(probabilities, 1).view(-1)
                if neox_args.return_logits:
                    generation_logits[step] = generated_token_logits[0]

            if neox_args.is_pipe_parallel or decode_tp_size > 1:
                if not sample_here:
                    generated_tokens = context_tokens.new_empty(batch_size)
                torch.distributed.broadcast(
                    generated_tokens, src=sample_src, group=sample_group
                )
            if neox_args.is_pipe_parallel and neox_args.return_logits:
                torch.distributed.broadcast(generation_logits[step], src=sample_src)

            # determine if state has started for each batch item
            state_started = (
                token_generation_start_index <= token_index_to_generate
            )  # check which batch items have been started

            # switch out padding tokens for generated tokens
            context_tokens[:, token_index_to_generate] = switch(
                context_tokens[:, token_index_to_generate].view(-1),
                generated_tokens,
                state_started,
            )

            # determine if state has finished for each batch item
            state_done = (
                generated_tokens == eos_token_id
            ).byte() & state_started.byte()  # check which batch items produce an eos_token in the current iteration
            state_is_done = state_is_done | state_done
            if stop_tokens:
                state_is_done |= _generated_stop_mask(
                    stop_tokens,
                    context_tokens,
                    token_index_to_generate + 1,
                    token_generation_start_index,
                )

            token_generation_end_index[
                (state_started.byte() & ~state_is_done).bool()
            ] = token_index_to_generate

            token_index_to_generate += 1

            scores = generation_logits[: step + 1] if neox_args.return_logits else None
            yield context_tokens, token_generation_start_index, token_generation_end_index, scores, state_is_done.bool()
            if torch.all(state_is_done):
                break


def generate_samples_from_prompt(
    neox_args,
    model,
    text: Union[List[str], str],
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    stop_tokens=None,
    minimum_tokens: int = 0,
    min_p: float = 0.0,
    top_h: float = 0.0,
    typical_p: float = 1.0,
    epsilon_cutoff: float = 0.0,
    eta_cutoff: float = 0.0,
):
    """Generate one completion per text prompt, sequentially and in input order.

    text may be a string or a list of strings. An empty string starts from EOS.
    See stream_tokens for decoding settings, cache setup and stop semantics.

    Returns a list of records on the prompt owner (global rank zero with
    pipeline parallelism, otherwise tensor-parallel rank zero); peers return [].
    Each record contains context, text (completion only, or None), length in
    tokens, finished, message and duration_seconds. If return_logits is enabled,
    logits contains one processed score row per executed step, including EOS;
    zero steps produces an empty list.
    """
    generate = _prompt_generator(
        neox_args,
        model,
        eos_token_id=eos_token_id,
        recompute=recompute,
        stop_tokens=stop_tokens,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    return generate(text)


def _prompt_generator(neox_args, model, *, eos_token_id=None, **options):
    if eos_token_id is None:
        eos_token_id = neox_args.tokenizer.eod
    stream = _prepare_token_stream(
        neox_args, model, eos_token_id=eos_token_id, **options
    )
    return partial(_generate_prompts, neox_args, model, eos_token_id, stream)


def _generate_prompts(neox_args, model, eos_token_id, stream, text):
    assert isinstance(text, (str, list)), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]
    input_count = len(text)
    input_pos = 0
    owns_prompts = (
        torch.distributed.get_rank() == 0
        if neox_args.is_pipe_parallel
        else is_mp_rank_0()
    )

    # generate completions
    generated_texts = []
    while True:

        start_time = time.time()
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if not owns_prompts:
            context_tokens = [eos_token_id]  # replaced by the prompt owner's tokens
        elif input_pos == input_count:
            terminate_runs = 1
        else:
            raw_text = text[input_pos]
            input_pos += 1

            if raw_text == "":
                context_tokens = [eos_token_id]
            else:
                context_tokens = neox_args.tokenizer.tokenize(raw_text)
            context_length = len(context_tokens)

            if context_length >= (neox_args.seq_length // 2):
                print_rank_0(
                    "\nWarning! Context length",
                    context_length,
                    "\nPlease give smaller context (e.g. half of the "
                    "max sequence length)!",
                )
        terminate_runs = broadcast_terminate_signal(
            terminate_runs, device=model.device, across_job=neox_args.is_pipe_parallel
        )
        if terminate_runs == 1:
            return generated_texts

        last_step = None
        for last_step in stream([context_tokens]):
            pass  # finish generation and use all results below

        if not owns_prompts:
            continue
        generated_tokens, finished, scores = [], False, None
        if last_step is not None:
            tokens, start, end, scores, done = last_step
            # This wrapper streams one prompt at a time.
            generated_tokens = tokens[0, start.item() : end.item() + 1].tolist()
            finished = done.item()

        generated_text, message = None, None
        if generated_tokens:
            try:
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
            except KeyError:
                message = "WARNING: generated token which doesn't exist."
        elif last_step is None:
            message = "No decode steps: the token budget or available context is zero."
        else:
            message = "WARNING: text generation did not start; try different batching or adjust parameters"
        data = {
            "context": raw_text,
            "text": generated_text,
            "length": len(generated_tokens),
            "finished": finished,
            "message": message,
            "duration_seconds": float(time.time() - start_time),
        }
        if neox_args.return_logits:
            data["logits"] = scores.tolist() if scores is not None else []
        generated_texts.append(data)

    return generated_texts


def generate_samples_input_from_file(
    neox_args,
    model,
    input_file,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    prompt_end: str = "\n",
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    minimum_tokens: int = 0,
    min_p: float = 0.0,
    top_h: float = 0.0,
    typical_p: float = 1.0,
    epsilon_cutoff: float = 0.0,
    eta_cutoff: float = 0.0,
):
    """Generate one record per nonempty prompt in a UTF-8 file.

    Split input_file at prompt_end (newline by default), strip whitespace, and
    process the remaining prompts in order as in generate_samples_from_prompt.
    Global rank zero overwrites output_file with JSONL. If this Python argument
    is None, the filename is input_file + ".output.jsonl". Returns the same list
    of records as generate_samples_from_prompt on the prompt owner, [] on peers.
    """
    generate = _prompt_generator(
        neox_args,
        model,
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    return _generate_from_file(generate, input_file, output_file, prompt_end)


def _generate_from_file(generate, input_file, output_file, prompt_end):
    print_rank_0(
        "generate_samples_input_from_file() loading input from {}".format(input_file)
    )
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = f.read()
        prompts = prompts.split(prompt_end)
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0(
        "generate_samples_input_from_file() prompts loaded: {}".format(len(prompts))
    )

    if output_file is None:
        output_file = str(input_file) + ".output.jsonl"
        print_rank_0(
            "generate_samples_input_from_file() setting default output file to {}".format(
                output_file
            )
        )

    print_rank_0("generate_samples_input_from_file() generating...")
    generated_texts = generate(prompts)

    if torch.distributed.get_rank() == 0:
        with open(output_file, "w") as f_out:
            for item in generated_texts:
                f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_input_from_file() done")
    return generated_texts


def generate_samples_unconditional(
    neox_args,
    model,
    number_of_samples: int = 10,
    output_file=None,
    eos_token_id: int = None,
    maximum_tokens: int = 64,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    minimum_tokens: int = 0,
    min_p: float = 0.0,
    top_h: float = 0.0,
    typical_p: float = 1.0,
    epsilon_cutoff: float = 0.0,
    eta_cutoff: float = 0.0,
):
    """Generate number_of_samples completions from an EOS seed, sequentially.

    number_of_samples must be positive. Return the records described by
    generate_samples_from_prompt on the prompt owner and [] on peers. If
    output_file is supplied, global rank zero overwrites it with one JSON
    record per completion.
    """

    generate = _prompt_generator(
        neox_args,
        model,
        eos_token_id=eos_token_id,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        recompute=recompute,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    return _generate_unconditional(generate, number_of_samples, output_file)


def _generate_unconditional(generate, number_of_samples, output_file):
    print_rank_0("generate_samples_unconditional() generating...")
    assert number_of_samples > 0, "number_of_samples must be > 0"
    generated_texts = generate([""] * number_of_samples)
    if output_file is not None and torch.distributed.get_rank() == 0:
        with open(output_file, "w") as f_out:
            for item in generated_texts:
                f_out.write(json.dumps(item) + "\n")
    print_rank_0("generate_samples_unconditional() done")
    return generated_texts


def generate_samples_interactive(
    neox_args,
    model,
    maximum_tokens: int = 64,
    prompt_end: str = "\n",
    eos_token_id: int = None,
    recompute: bool = False,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    minimum_tokens: int = 0,
    min_p: float = 0.0,
    top_h: float = 0.0,
    typical_p: float = 1.0,
    epsilon_cutoff: float = 0.0,
    eta_cutoff: float = 0.0,
):
    """Read terminal prompts on global rank zero and print the growing completion.

    prompt_end delimits each prompt (newline by default). All ranks participate
    in decoding; EOF ends the session. Returns None and does not write a file.
    See stream_tokens for the decoding options.
    """

    stream = _prepare_token_stream(
        neox_args,
        model,
        eos_token_id=eos_token_id,
        recompute=recompute,
        maximum_tokens=maximum_tokens,
        minimum_tokens=minimum_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        top_h=top_h,
        typical_p=typical_p,
        epsilon_cutoff=epsilon_cutoff,
        eta_cutoff=eta_cutoff,
    )
    return _generate_interactive(neox_args, stream, prompt_end)


def _generate_interactive(neox_args, stream, prompt_end):
    reader = torch.distributed.get_rank() == 0
    first_prompt = True
    while True:
        prompt = [None]  # None signals all stages to exit together.
        if reader:
            try:
                if not first_prompt:
                    input("\n<press enter to continue>")
                os.system("clear")
                lines = []
                while True:
                    line = input("Context prompt >>> ")
                    if prompt_end == "\n" or prompt_end in line:
                        lines.append(line.split(prompt_end, 1)[0])
                        break
                    lines.append(line)
                tokens = neox_args.tokenizer.tokenize("\n".join(lines))
                if len(tokens) >= neox_args.seq_length - 1:
                    print_rank_0("Reached max sequence length!")
                else:
                    prompt[0] = tokens if len(tokens) else [neox_args.tokenizer.eod]
            except EOFError:
                pass
        torch.distributed.broadcast_object_list(prompt, src=0)
        if prompt[0] is None:
            return
        first_prompt = False
        for (
            batch_context_tokens,
            batch_token_generation_start_index,
            batch_token_generation_end_index,
            batch_generated_token_logits,
            is_done,
        ) in stream(prompt):
            if reader:
                generated_tokens = (
                    batch_context_tokens[0]
                    .cpu()
                    .numpy()
                    .tolist()[
                        batch_token_generation_start_index[0]
                        .item() : batch_token_generation_end_index[0]
                        .item()
                        + 1
                    ]
                )
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                print_rank_0("Generated Text: " + generated_text)


def generate_from_config(neox_args, model):
    """Run the CLI mode using only the options validated by NeoXArgs at load time."""
    mode = neox_args.text_gen_type
    if mode == "precompute":
        return precompute_logits(neox_args, model)
    if mode not in ("interactive", "input-file", "unconditional"):
        raise ValueError(
            f"`text_gen_type` either not specified or not recognised: {mode}"
        )
    stream = _bind_token_stream(
        neox_args,
        model,
        recompute=neox_args.recompute,
        **sampling_options_from_config(neox_args),
    )
    if mode == "interactive":
        return _generate_interactive(neox_args, stream, neox_args.prompt_end)
    generate = partial(
        _generate_prompts, neox_args, model, neox_args.tokenizer.eod, stream
    )
    if mode == "input-file":
        assert neox_args.sample_input_file is not None
        print_rank_0(
            f"Generating samples from input file {neox_args.sample_input_file}"
        )
        return _generate_from_file(
            generate,
            neox_args.sample_input_file,
            neox_args.sample_output_file,
            neox_args.prompt_end,
        )
    print_rank_0(
        f"Generating samples unconditionally and saving results to {neox_args.sample_output_file}"
    )
    return _generate_unconditional(
        generate, neox_args.num_samples, neox_args.sample_output_file
    )


def get_logp(logits, labels, force_fp32=False):
    if force_fp32:
        logits = logits.float()
    logp = logits.log_softmax(dim=-1)
    return torch.gather(logp, dim=2, index=labels.unsqueeze(2)).squeeze(2)


def precompute_logits(neox_args, model):
    """
    Precomputes logprobs from training/testing/validation datasets

    Saves it to the same directory as the dataset with the model name appended to it

    neox_args: NeoXArgs.
    model: a Megatron model

    """
    if neox_args.precompute_model_name is None:
        mdl_name = str(hash(neox_args.load))
    else:
        mdl_name = neox_args.precompute_model_name
    print_rank_0("Precomputing logprobs...")
    model.eval()
    data_paths = list()
    if neox_args.train_data_paths is not None:
        for path in neox_args.train_data_paths:
            data_paths.append(path)
        for path in neox_args.test_data_paths:
            data_paths.append(path)
        for path in neox_args.valid_data_paths:
            data_paths.append(path)
    elif neox_args.pos_train_data_paths is not None:
        # Pairwise data...
        for path in neox_args.pos_train_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_train_data_paths:
            data_paths.append(path)
        for path in neox_args.pos_valid_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_valid_data_paths:
            data_paths.append(path)
        for path in neox_args.pos_test_data_paths:
            data_paths.append(path)
        for path in neox_args.neg_test_data_paths:
            data_paths.append(path)
    for path in data_paths:
        print_rank_0(f"Precomputing logits for {path}")
        # Add hash to path...
        out_path = path + f"_{mdl_name}"
        if os.path.exists(out_path + ".idx"):
            continue
        dataset = make_dataset(path, neox_args.data_impl, not neox_args.mmap_warmup)
        if is_mp_rank_0():
            out_dataset = make_builder(out_path + ".bin", neox_args.data_impl)
            out_dataset._dtype = np.float32
        i = 0

        # TODO: Not sure why this requires a multiple of 8? Investigate later.
        while i < int(math.ceil(len(dataset) / 8.0) * 8):
            start = time.time()
            model.module.clear_cache()  # clear kv cache between batches
            if is_mp_rank_0():
                offset = (
                    mpu.get_data_parallel_rank()
                    * neox_args.train_micro_batch_size_per_gpu
                )
                context_tokens = [
                    [int(x) for x in dataset.get(j % len(dataset)).tolist()]
                    for j in range(
                        i + offset,
                        i + (neox_args.train_micro_batch_size_per_gpu + offset),
                    )
                ]
                # grab microbatch
                # pad batch in order to allow conversion to tensor
                context_tokens, context_lengths = pad_batch(
                    copy.deepcopy(context_tokens),
                    pad_id=0,
                    pad_len=neox_args.seq_length + 1,
                    truncate=True,
                )
                # print(context_tokens)
                label_tokens = [tokens[1:] for tokens in context_tokens]
                context_tokens = [tokens[:-1] for tokens in context_tokens]
            else:
                context_tokens = [
                    [0 for _ in range(neox_args.seq_length)]
                    for _ in range(neox_args.batch_size)
                ]
                label_tokens = [
                    [0 for _ in range(neox_args.seq_length)]
                    for _ in range(neox_args.batch_size)
                ]
                context_lengths = [0 for _ in range(neox_args.batch_size)]
            i += (
                neox_args.train_micro_batch_size_per_gpu
                * mpu.get_data_parallel_world_size()
            )
            # print(context_tokens)
            # convert to tensor and broadcast
            context_tokens = torch.cuda.LongTensor(context_tokens)
            label_tokens = torch.cuda.LongTensor(label_tokens)
            # Make sure context tokens + start tokens are the same across all ranks
            token_generation_start_index = torch.cuda.LongTensor(context_lengths)
            torch.distributed.broadcast(
                context_tokens,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            torch.distributed.broadcast(
                token_generation_start_index,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            torch.distributed.broadcast(
                label_tokens,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group(),
            )
            # context_tokens = context_tokens[:, :chop_len].contiguous()
            # label_tokens = label_tokens[:, :chop_len].contiguous()
            with torch.no_grad():
                # get attention mask / position ids
                context_tokens, attention_mask, position_ids = get_batch(
                    neox_args, context_tokens
                )
                model_inputs = (
                    context_tokens,
                    position_ids,
                    attention_mask,
                )
                maybe_tuple = forward_model(
                    model, model_inputs, neox_args.is_pipe_parallel
                )
                if isinstance(maybe_tuple, tuple):
                    logits, _ = maybe_tuple
                else:
                    logits = maybe_tuple
                if logits is not None:  # if pipe parallel, not all ranks return logits
                    logits = gather_from_model_parallel_region(logits)
                    logp = get_logp(logits, label_tokens, True).squeeze()
                if neox_args.is_pipe_parallel:
                    # broadcast generated tokens to pipe parallel group
                    src_rank = model.grid.stage_to_global(model.num_stages - 1)
                    logp = (
                        logp
                        if logits is not None
                        else torch.zeros(
                            neox_args.batch_size, dtype=torch.float32
                        ).cuda()
                    )
                    torch.distributed.broadcast(
                        tensor=logp,
                        src=src_rank,
                        group=mpu.get_pipe_parallel_group(),
                    )
                    logp = logp.squeeze()
                logp_list = [
                    torch.zeros_like(logp)
                    for _ in range(mpu.get_data_parallel_world_size())
                ]
                torch.distributed.all_gather(
                    logp_list, logp, group=mpu.get_data_parallel_group()
                )
                logp = torch.cat(logp_list, dim=0).cpu().numpy()
                if (mpu.get_model_parallel_rank() == 0) and (
                    mpu.get_data_parallel_rank() == 0
                ):
                    for j in range(logp.shape[0]):
                        out_dataset.add_item(logp[j])
                        out_dataset.end_document()
            print_rank_0(f"Processed {i} / {len(dataset)} in {time.time() - start}")
        if is_mp_rank_0():
            out_dataset.finalize(
                out_path + ".idx",
            )
        torch.distributed.barrier()
