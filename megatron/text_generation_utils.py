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
from megatron.utils import get_ltor_masks_and_position_ids


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

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
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


def generate_samples_input_from_file(model):
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.get_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print_rank_0('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    ctr = 0
    while True:
        start_time = time.time()
        # Convert text into tokens
        # torch.distributed.barrier(group=mpu.get_model_parallel_group())
        # terminate_runs = 0

        terminate_runs = 0
        if mpu.get_model_parallel_rank() == 0:
            raw_text = all_raw_text[input_pos]
            input_pos += 1
            if input_pos == input_count:
                terminate_runs = 1

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

        # Send signal to all workers to terminate if
        terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
        torch.distributed.broadcast(terminate_runs_tensor,
                                    mpu.get_model_parallel_src_rank(),
                                    group=mpu.get_model_parallel_group())
        terminate_runs = terminate_runs_tensor[0].item()

        if terminate_runs == 1:
            return

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

            if mpu.get_model_parallel_rank() == 0:
                datum = {'context': raw_text, 'text': text, 'length': length - 1, 'finished': is_finished}
                fname_out.write(json.dumps(datum) + '\n')

                print_rank_0("\nYou say:")
                print_rank_0(raw_text)
                print_rank_0("\nI say:")
                print_rank_0(text)

                if ctr % args.log_interval == 0:
                    print_rank_0('Avg s/batch:',
                          (time.time() - start_time) / min(args.log_interval, ctr + 1))
                    start_time = time.time()
                ctr += 1


def generate_samples_interactive(model, print_frequency=24):

    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if mpu.get_model_parallel_rank() == 0:
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

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print_rank_0("\nContext:", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[len(raw_text):]
                print_rank_0("\nMegatron-LM:", trim_decode_tokens, flush=True)

            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

            if mpu.get_model_parallel_rank() == 0:
                input("\nPress any key to continue >>>")


def generate_samples_unconditional(model):

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


def pad_batch(batch, pad_id, args):

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens):

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

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):

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
                logits = model(tokens,
                               position_ids,
                               attention_mask,
                               tokentype_ids=type_ids,
                               forward_method_parallel_output=False)
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
                logits, layer_past = model(tokens2use,
                                           positions2use,
                                           attention_mask,
                                           layer_past=layer_past,
                                           get_key_value=True,
                                           tokentype_ids=types2use,
                                           forward_method_parallel_output=False)
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
