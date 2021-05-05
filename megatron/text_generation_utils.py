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
from typing import List, Union

import torch
import torch.nn.functional as F

from megatron import print_rank_0
from megatron import mpu
from megatron.utils import get_ltor_masks_and_position_ids, is_mp_rank_0


def get_batch(neox_args, context_tokens: torch.Tensor):
    """
    Generate batch from context tokens. Attention mask and position ids are created. Returned tensors will be on CUDA.
    
    neox_args: instantiated NeoXArgs with tokenizer instantiated and reset_position_ids, reset_attention_mask and eod_mask_loss defined
    context_tokens: torch tensor with dimensions [batch, context_size]

    returns: tuple of torch tensors (tokens, attention_mask, position_ids) on CUDA
    """
 
    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        neox_args.tokenizer.eod,
        neox_args.reset_position_ids,
        neox_args.reset_attention_mask,
        neox_args.eod_mask_loss)
    return tokens, attention_mask, position_ids

def pad_batch(context_tokens: List[List[int]], pad_id: int, pad_len: int):
    """
    pads context lengths in context_tokens with pad_id to equal neox_args.seq_length,
    and returns the padded batch and the new lengths.

    context_tokens: list of lists of tokens
    pad_id: int, integer to use as padding token
    pad_len: int, context length to be padded; all batch items will be padded to the same length

    returns: tuple of padded context tokens and a list of unpadded token count
    """

    context_lengths = []
    for tokens in context_tokens:
        context_length = len(tokens)
        if context_length < pad_len:
            tokens.extend([pad_id] * (pad_len - context_length))
        elif context_length > pad_len:
            raise ValueError("context_length is bigger than to be padded length")
        context_lengths.append(context_length)
    return context_tokens, context_lengths

def filter_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
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
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits

def switch(val1, val2, boolean):
    """
    replaces items in val1 with items in val2 where boolean = True
    """
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2

def forward_model(neox_args, model, model_inputs):
    """
    Runs model.forward(model_inputs)

    We need to create a wrapper for this function because deepspeed pipe parallel modules operate differently to normal models.

    model: a Megatron model.
    model_inputs: tuple containing model args

    returns: result of model.forward(model_inputs)
    """
    # because someone at deepspeed decided pipeline modules couldn't use kwargs,
    # we need to forward a pipe model by access model.module() instead of just model()

    torch.distributed.barrier()
    if neox_args.pipe_parallel_size <= 1:
        return model.module(model_inputs)
    else:
        data_iterator = iter(
            [[model_inputs, torch.Tensor(1)]])  # we need to feed in fake labels bc deepspeed is only built for training
        x = model.inference_batch(data_iterator)
        return x

def broadcast_terminate_signal(terminate_runs: int):
    """Send signal to all workers to terminate if we've finished the process"""
    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(terminate_runs_tensor,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    return terminate_runs_tensor[0].item()

def get_token_stream(neox_args, model, context_tokens: List[List[int]], eos_token_id: int = None, max_tokens: int = None, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    iterator producing text completions

    neox_args: instantiated NeoXArgs with instantiated tokenizer 
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids

    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    max_tokens: maximum number of tokens to be generated; careful! if a batch input is provided max_tokens specifies the maximum number of forwards. longer batch items get less generated tokens.

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucles) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: tokens (completions from model), token_generation_start_index (token index per batch item for the first generated token), token_generation_end_index (token index per batch item for the last generated token)
            * each iteration adds a generated token to the context_tokens
            * output contains both context_tokens from input and generated tokens
            * if batch items have different lengths, the iterator will start at the first completion and return the unchanged input context token otherwise
    """

    model.eval()

    # pad batch in order to allow conversion to tensor
    context_tokens, context_lengths = pad_batch(copy.deepcopy(context_tokens), pad_id=neox_args.tokenizer.eod, pad_len=neox_args.seq_length) 

    # convert to tensor and broadcast
    context_tokens = torch.cuda.LongTensor(context_tokens)
    token_generation_start_index = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(context_tokens,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(token_generation_start_index,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    
    # produce batch relevant attention_mask and position_ids
    tokens, attention_mask, position_ids = get_batch(neox_args, context_tokens)

    # determine the smallest context length at which first output is produced
    context_length = token_generation_start_index.min().item()

    # set variables
    eos_token_id = eos_token_id or neox_args.tokenizer.eod
    max_tokens = max_tokens or (neox_args.seq_length - token_generation_start_index.max().item() - 1)
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length - 1, # never generate more than the model's sequence length
        token_index_to_generate + max_tokens -1
    )

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        layer_past = torch.Tensor().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute or (token_index_to_generate == first_token_index_to_generate):
                # when recomputing or at first iteration all tokens are forwarded
                tokens_to_use = context_tokens[:, :token_index_to_generate]
                positions_to_use = position_ids[:, :token_index_to_generate]
            else:
                # otherwise only the last tokens are forwarded and layer past is used for other tokens
                tokens_to_use = context_tokens[:, token_index_to_generate - 1].view(batch_size, -1) # view applied to keep dimensions
                positions_to_use = position_ids[:, token_index_to_generate - 1].view(batch_size, -1)
            
            # we have to use a tuple instead of kwargs here because deepspeed :|
            model_inputs = (
                tokens_to_use,
                positions_to_use,
                attention_mask,
                layer_past,
                )

            logits, layer_past = forward_model(neox_args, model, model_inputs)
            
            # TODO: we are replicating computation across all machines here, which is really unecessary,
            #  we should probably just do it on one then communicate the results?
            
            # sample token id of the to be generated token
            generated_token_logits = logits[:, -1].view(batch_size, -1).contiguous()
            if temperature == 0.0 and top_k == 0 and top_p == 0.0:
                generated_tokens = torch.argmax(generated_token_logits, dim=-1).view(-1)
            else:
                generated_token_logits = generated_token_logits.float()
                if temperature > 0.0:
                    generated_token_logits /= temperature
                generated_token_logits = filter_logits(generated_token_logits, top_k=top_k, top_p=top_p)
                next_token_log_probs = F.softmax(generated_token_logits, dim=-1)
                generated_tokens = torch.multinomial(next_token_log_probs, num_samples=1).view(-1)

        	# determine state for each batch item
            state_started = token_generation_start_index <= token_index_to_generate # check which batch items have been started
            state_done = (generated_tokens == eos_token_id).byte() & state_started.byte() # check which batch items produce an eos_token in the current iteration
            state_just_finished = (state_done & ~state_is_done).bool()
            state_is_done = state_is_done | state_done
            token_generation_end_index[(state_started.byte() & ~state_is_done).bool()] = token_index_to_generate

            # switch out only padding tokens (the batch items that have been started)
            context_tokens[:, token_index_to_generate] = switch(context_tokens[:, token_index_to_generate].view(-1), generated_tokens, state_started) 
            
            token_index_to_generate += 1
            
            
            yield context_tokens, token_generation_start_index, token_generation_end_index
            if torch.all(state_is_done): break

def generate_samples_from_prompt(neox_args, model, text: Union[List[str], str], eos_token_id: int = None, max_tokens: int = 64, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generates samples from raw text and returns them in a dictionary.


    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    max_tokens: maximum number of tokens to be generated; careful! if a batch input is provided max_tokens specifies the maximum number of forwards. longer batch items get less generated tokens.

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucles) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished': 
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds 
        
    """

    # type check
    assert any([isinstance(text, str), isinstance(text, list)]), "Text should be in string or list form"
    if isinstance(text, str):
        text = [text]

    if is_mp_rank_0():
        input_count = len(text)
        input_pos = 0

    # generate completions
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

                context_tokens = neox_args.tokenizer.tokenize(raw_text)
                context_length = len(context_tokens)

                if context_length >= (neox_args.seq_length // 2):
                    print_rank_0("\nContext length", context_length,
                                 "\nPlease give smaller context (half of the "
                                 "sequence length)!", flush=True)
                    continue
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts

        for batch_context_tokens, batch_token_generation_start_index, batch_token_generation_end_index in get_token_stream(
            neox_args=neox_args, 
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=neox_args.tokenizer.eod,
            max_tokens=max_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
            ):
            pass # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = batch_token_generation_start_index.cpu().numpy().tolist()
        batch_token_generation_end_index = batch_token_generation_end_index.cpu().numpy().tolist()
        for tokens, start_index, end_index in zip(batch_context_tokens, batch_token_generation_start_index, batch_token_generation_end_index):
            if end_index >= start_index:
                generated_tokens = tokens[start_index:end_index + 1]
                try:
                    generated_text = neox_args.tokenizer.detokenize(generated_tokens)
                    message = None
                except KeyError:
                    generated_text = None
                    message = "WARNING: generated token which doesn't exist."
            else:
                generated_tokens = list()
                message = "WARNING: text generation did not start; try different batching or adjust parameters"
            is_finished = (end_index < neox_args.seq_length - 1) and end_index > -1
            if is_mp_rank_0():
                data = {'context': raw_text, 'text': generated_text, 'length': len(generated_tokens), 'finished': is_finished, 'message': message, 'duration_seconds': float(time.time() - start_time)}
                generated_texts.append(data)
                
    return generated_texts


def generate_samples_input_from_file(neox_args, model):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file

    model: a Megatron model
    """
    # Read the sample file and open the output file.
    assert neox_args.sample_input_file is not None, \
        'sample input file is not provided.'
    with open(neox_args.sample_input_file, "r") as f:
        prompts = f.readlines()
    if is_mp_rank_0():
        if neox_args.sample_output_file is None:
            sample_output_file = neox_args.sample_input_file + ".out"
            print_rank_0('could not find `sample-output-file`, setting '
                         'it to {}'.format(sample_output_file))
        else:
            sample_output_file = neox_args.sample_output_file
        f_out = open(sample_output_file, "w+")
    generated_texts = generate_samples_from_prompt(neox_args=neox_args, model=model, text=prompts)
    if is_mp_rank_0():
        for item in generated_texts:
            f_out.write(json.dumps(item) + '\n')


def generate_samples_interactive(neox_args, model, print_frequency=24):
    """
    Generates samples interactively in the terminal.

    model: a Megatron model
    print_frequency: int, how often (in tokens) to print the output.
    """

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
                    context_tokens = neox_args.tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (neox_args.seq_length // 2):
                        print_rank_0("\nContext length", context_length,
                                     "\nPlease give smaller context (half of the "
                                     "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs = broadcast_terminate_signal(terminate_runs)
            if terminate_runs == 1:
                return

            token_stream = get_token_stream(neox_args, model, [context_tokens])
            for counter, decode_tokens in enumerate(token_stream):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

                if mpu.get_model_parallel_rank() == 0 and \
                        counter % print_frequency == 0:
                    os.system('clear')
                    print_rank_0("\nContext:", raw_text, flush=True)
                    trim_decode_tokens = neox_args.tokenizer.detokenize(
                        decode_tokens)[len(raw_text):]
                    print_rank_0("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if is_mp_rank_0():
                os.system('clear')
                print_rank_0("\nContext:", raw_text, flush=True)
                trim_decode_tokens = neox_args.tokenizer.detokenize(
                    decode_tokens)[len(raw_text):]
                print_rank_0("\nMegatron-LM:", trim_decode_tokens, flush=True)

            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

            if is_mp_rank_0():
                input("\nPress any key to continue >>>")


def generate_samples_unconditional(neox_args, model):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.


    model: a Megatron model

    yields: Dict -> a dict containing the following fields:
        - 'text' (the completion)
        - 'length' (the length of the completion)
    """

    num_samples = neox_args.num_samples
    context_tokens = [[neox_args.tokenizer.eod]
                      for _ in range(neox_args.batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        token_stream = None

        for token_stream in get_token_stream(neox_args, model, copy.deepcopy(context_tokens)):
            ' print(token_stream) -> 1, 1,2, 1,2,3'
            pass
        if token_stream is None: break
        if ctr % neox_args.log_interval == 0:
            print_rank_0('Avg s/batch:',
                         (time.time() - start_time) / min(neox_args.log_interval, ctr + 1))
            start_time = time.time()
        length = len(token_stream)
        token_batch = token_stream[0].cpu().numpy().tolist()
        length_batch = token_stream[1].cpu().numpy().tolist()

        for tokens, length in zip(token_batch, length_batch):
            tokens = tokens[1:length - 1]
            try:
                text = neox_args.tokenizer.detokenize(tokens)
            except KeyError:
                print_rank_0("WARNING: generated token which doesn't exist. Skipping")
                continue
            is_finished = length < neox_args.seq_length - 1
            datum = {'text': text, 'length': length - 1, 'finished': is_finished}
            yield datum
            ctr += 1
            if ctr >= num_samples:
                break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(neox_args, model):
    """
    Generates samples unconditionially (no prompt) and writes them to an output file at neox_args.genfile

    model: a Megatron model

    """
    assert neox_args.genfile is not None
    genfile = neox_args.genfile

    # Create directory
    genfile_dir = os.path.dirname(genfile)
    os.makedirs(genfile_dir, exist_ok=True)

    with open(genfile, 'w') as f:
        for n, datum in enumerate(generate_samples_unconditional(neox_args=neox_args, model=model), 1):
            f.write(json.dumps(datum) + '\n')
            if n != 0 and n % neox_args.log_interval:
                print_rank_0(f"Text generated and written: {n}")
