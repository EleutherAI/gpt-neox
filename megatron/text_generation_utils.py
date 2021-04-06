# coding=utf-8
# Copyright (c) 2021  Josh Levy-Kramer <josh@levykramer.co.uk>. All rights reserved.
# This file is based on code by the authors denoted below and has been modified from its original version.
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

"""Utilities for generating text."""

import copy
import json
import os
import time

import torch
import torch.nn.functional as F

from megatron import get_args, print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0
from megatron.fp16 import fp32_to_fp16
from typing import List, Union

def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.pipe_parallel_size >= 1 and args.fp16:
        # cast to fp16 because pipeline parallelism skips the FP16 wrapper.
        return fp32_to_fp16((tokens, attention_mask, position_ids))
    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 
    Filters the logits using top_k / top_p, filling any filtered vocab items with filter_value (defaults to -inf).

    This function has been mostly taken from huggingface conversational ai code at
    https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
    
    logits: torch.Tensor -> logits of megatron model.
    top_k: integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token. 
    top_p: float -> Top-p (nucles) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p. 
    
    returns: (filtered) logits"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def pad_batch(batch, pad_id, args):
    """
    pads context lengths in batch with pad_id to equal args.seq_length,
    and returns the padded batch and the new lengths.

    batch: torch.Tensor of tokens
    pad_id: int, integer to use as padding token
    args: global args
    """

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens):
    """
    yields completions from a model as an iterator.

    model: a Megatron model.
    context_tokens: the prompt to complete.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(context_length_tensor,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        yield tokens[:, :context_length], lengths


def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_model(model, model_inputs):
    """
    Runs model.forward(model_inputs)

    We need to create a wrapper for this function because deepspeed pipe parallel modules operate differently to normal models.

    model: a Megatron model.
    model_inputs: tuple containing model args

    returns: result of model.forward(model_inputs)
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model by access model.module() instead of just model()
    args = get_args()
    if args.pipe_parallel_size == 1:
        return model.module(model_inputs)
    elif args.pipe_parallel_size > 1:
        # deepspeed makes this super difficult
        raise NotImplementedError
        # data_iterator = iter([[model_inputs, torch.Tensor(1)]])
        # train_batch_fn = model.batch_fn
        # model.set_batch_fn(lambda x: x)
        # x = model.eval_batch(data_iterator)
        # model.set_batch_fn(train_batch_fn)
        # return x
    else:
        return model(*model_inputs)


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):
    """
    yields completions from a model as an iterator.

    model: a Megatron model.
    context_tokens: the prompt to complete.
    context_lengths: lengths of context tokens. 
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.

    yields: tokens (completions from model), and lengths (lengths of completions)
    """
    args = get_args()
    tokenizer = get_tokenizer()

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length <= (maxlen):

            if args.recompute:
                # we need to use args instead of kwargs here because deepspeed :|
                model_inputs = (tokens,
                               position_ids,
                               attention_mask,
                               type_ids, # tokentype_ids
                               None, # layer_past,
                               False, # get_key_value
                               )
                logits = forward_model(model, model_inputs)
                logits = logits[:, context_length - 1, :]
            else:
                types2use = None
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                    if type_ids is not None:
                        types2use = type_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                    if type_ids is not None:
                        types2use = type_ids[:, context_length - 1].view(
                            batch_size, -1)
                # we have to use args instead of kwargs here because deepspeed :|
                model_inputs = (tokens2use, # input_ids
                                positions2use, # position_ids
                                attention_mask, # attention_mask
                                types2use, # tokentype_ids
                                layer_past, # layer_past
                                True, # get_key_value
                                )
                logits, layer_past = forward_model(model, model_inputs)
                logits = logits[:, -1].view(batch_size, -1).contiguous()

            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k,
                                      top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            print_logits = []
            for p in prev:
                print_logits.append([logits[i, p].item()
                                     for i in range(batch_size)])
            started = context_lengths <= context_length
            tokens[:, context_length] = switch(
                tokens[:, context_length].view(-1), prev, started)
            context_length += 1
            counter += 1

            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token
            done = torch.all(is_done)

            yield tokens, lengths
            if done:
                break


def generate_samples_from_prompt(model, text: Union[List[str], str]):
    """
    Generates samples from raw text and returns them in a dictionary.


    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    returns: List[dict] -> a list of dicts containing the following fields: 
        - 'context' (the input) 
        - 'text' (the completion) 
        - 'length' (the length of the completion)
    """
    args = get_args()
    tokenizer = get_tokenizer()

    # type check
    assert any([isinstance(text, str), isinstance(text, list)]), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]
    
    if is_mp_rank_0():
        input_count = len(text)
        input_pos = 0

    # generate completions
    iterations = 0
    generated_texts = []
    while True:
        start_time = time.time()
        
        # Tokenize text, and check whether we should terminate process
        terminate_runs = 0
        if is_mp_rank_0():
            if input_pos == input_count:
                terminate_runs = 1
            else:
                raw_text = text[input_pos]
                input_pos += 1

                context_tokens = tokenizer.tokenize(raw_text)
                context_length = len(context_tokens)

                if context_length >= (args.seq_length // 2):
                    print_rank_0("\nContext length", context_length,
                        "\nPlease give smaller context (half of the "
                        "sequence length)!", flush=True)
                    continue
        else:
            context_tokens = tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        # Send signal to all workers to terminate if we've finished the process
        terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        torch.distributed.broadcast(terminate_runs_tensor,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        terminate_runs = terminate_runs_tensor[0].item()
        if terminate_runs == 1:
            return generated_texts

        for token_stream in get_token_stream(model, copy.deepcopy([context_tokens])):
            pass
        token_batch = token_stream[0].cpu().numpy().tolist()
        length_batch = token_stream[1].cpu().numpy().tolist()
        for tokens, length in zip(token_batch, length_batch):
            tokens = tokens[1:length - 1]
            try:
                text = tokenizer.detokenize(tokens)
            except KeyError:
                print_rank_0("WARNING: generated token which doesn't exist. Skipping")
                continue
            is_finished = length < args.seq_length - 1

            if is_mp_rank_0():
                data = {'context': raw_text, 'text': text, 'length': length - 1, 'finished': is_finished}
                generated_texts.append(data)
                if iterations % args.log_interval == 0:
                    print_rank_0('Avg s/batch:',
                          (time.time() - start_time) / min(args.log_interval, iterations + 1))
                    start_time = time.time()
                iterations += 1

    return generated_texts


def generate_samples_input_from_file(model):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from args.sample_input_file and writes completions to args.sample_output_file

    model: a Megatron model
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    with open(args.sample_input_file, "r") as f:
        prompts = f.readlines()
    if is_mp_rank_0():
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print_rank_0('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        f_out = open(sample_output_file, "w+")
    generated_texts = generate_samples_from_prompt(model, prompts)
    if is_mp_rank_0():
        for item in generated_texts:
            f_out.write(json.dumps(item) + '\n')


def generate_samples_interactive(model, print_frequency=24):
    """
    Generates samples interactively in the terminal.

    model: a Megatron model
    print_frequency: int, how often (in tokens) to print the output.
    """
    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if is_mp_rank_0():
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print_rank_0('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print_rank_0("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for counter, decode_tokens in enumerate(token_stream):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

                if mpu.get_model_parallel_rank() == 0 and \
                   counter % print_frequency == 0:
                    os.system('clear')
                    print_rank_0("\nContext:", raw_text, flush=True)
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[len(raw_text):]
                    print_rank_0("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if is_mp_rank_0():
                os.system('clear')
                print_rank_0("\nContext:", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[len(raw_text):]
                print_rank_0("\nMegatron-LM:", trim_decode_tokens, flush=True)

            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

            if is_mp_rank_0():
                input("\nPress any key to continue >>>")


def generate_samples_unconditional(model):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.


    model: a Megatron model

    yields: Dict -> a dict containing the following fields: 
        - 'text' (the completion) 
        - 'length' (the length of the completion)
    """
    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod]
                      for _ in range(args.batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        for token_stream in get_token_stream(model, copy.deepcopy(context_tokens)):
            pass
        if ctr % args.log_interval == 0:
            print_rank_0('Avg s/batch:',
                  (time.time() - start_time) / min(args.log_interval, ctr + 1))
            start_time = time.time()
        length = len(token_stream)
        token_batch = token_stream[0].cpu().numpy().tolist()
        length_batch = token_stream[1].cpu().numpy().tolist()

        for tokens, length in zip(token_batch, length_batch):
            tokens = tokens[1:length - 1]
            try:
                text = tokenizer.detokenize(tokens)
            except KeyError:
                print_rank_0("WARNING: generated token which doesn't exist. Skipping")
                continue
            is_finished = length < args.seq_length - 1
            datum = {'text': text, 'length': length - 1, 'finished': is_finished}
            yield datum
            ctr += 1
            if ctr >= num_samples:
                break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model):
    """
    Generates samples unconditionially (no prompt) and writes them to an output file at args.genfile

    model: a Megatron model

    """
    args = get_args()
    assert args.genfile is not None
    genfile = args.genfile

    # Create directory
    genfile_dir = os.path.dirname(genfile)
    os.makedirs(genfile_dir, exist_ok=True)

    with open(genfile, 'w') as f:
        for n, datum in enumerate(generate_samples_unconditional(model), 1):
            f.write(json.dumps(datum) + '\n')
            if n != 0 and n % args.log_interval:
                print_rank_0(f"Text generated and written: {n}")
