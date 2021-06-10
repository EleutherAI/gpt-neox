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
    
    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
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
    top_p: float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.

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

def stream_tokens(neox_args, model, context_tokens: List[List[int]], eos_token_id: int = None, 
                    maximum_tokens: int = None, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    iterator producing text completions

    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
    model: a Megatron model.
    context_tokens: the prompt to complete; unpadded list of lists of tokens ids

    context_lengths: lengths of context tokens of dimension [batch]; the context length records for each bach item how many non-padded tokens are provided
    attention_mask: attention mask for megatron model.
    position_ids: position ids for positional encoding.

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated; careful! if a batch input is provided maximum_tokens specifies the maximum number of forwards. 
                    longer batch items get less generated tokens.

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: (
                tokens (completions from model), 
                token_generation_start_index (token index per batch item for the first generated token), 
                token_generation_end_index (token index per batch item for the last generated token),
                logits (logits which are so far computed, zeros otherwise),
                is_done (flag for each bach item indicating whether an eod token was generated)
            )

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
    maximum_tokens = maximum_tokens or (neox_args.seq_length - token_generation_start_index.max().item() - 1)
    batch_size = context_tokens.size(0)

    # get the context_index at which generation is to start
    # we start generation at the position where the smallest context ends
    token_index_to_generate = token_generation_start_index.min().item()
    first_token_index_to_generate = token_index_to_generate
    last_token_index_to_generate = min(
        neox_args.seq_length - 1, # never generate more than the model's sequence length
        token_index_to_generate + maximum_tokens -1
    )

    all_logits = torch.zeros((batch_size, neox_args.seq_length, neox_args.padded_vocab_size))

    with torch.no_grad():
        # initialize generation variables
        state_is_done = torch.zeros([batch_size]).byte().cuda()
        layer_past = torch.Tensor().cuda()
        token_generation_end_index = torch.ones([batch_size]).long().cuda() * (-1)

        while token_index_to_generate <= last_token_index_to_generate:
            if recompute:
                # recompute is needed for sparse attention at the moment
                # because we can only forward multiples of the block size
                # TODO The full padded context_tokens would not need to be forwarded, adjust to multiples of block size
                # we need to use neox_args instead of kwargs here because deepspeed :|
                model_inputs = (context_tokens,
                                position_ids,
                                attention_mask,
                                torch.Tensor(),
                                )
                logits, _ = forward_model(neox_args, model, model_inputs)
                generated_token_logits = logits[:, token_index_to_generate - 1, :]
                all_logits = logits
            else:
                # not choosing recompute assumes that any number of tokens can be forwarded
                # this is not the case for sparse attention
                if token_index_to_generate == first_token_index_to_generate:
                    tokens_to_use = tokens[:, :token_index_to_generate]
                    positions_to_use = position_ids[:, :token_index_to_generate]
                else:
                    tokens_to_use = tokens[:, token_index_to_generate - 1].view(
                        batch_size, -1)
                    positions_to_use = position_ids[:, token_index_to_generate - 1].view(
                        batch_size, -1)
                # we have to use neox_args instead of kwargs here because deepspeed :|
                model_inputs = (tokens_to_use,  # input_ids
                                positions_to_use,  # position_ids
                                attention_mask,  # attention_mask
                                layer_past,  # layer_past
                                )

                logits, layer_past = forward_model(neox_args, model, model_inputs)
                # TODO: we are replicating computation across all machines here, which is really unecessary,
                #  we should probably just do it on one then communicate the results?
                generated_token_logits = logits[:, -1].view(batch_size, -1).contiguous()
                
                if token_index_to_generate == first_token_index_to_generate:
                    all_logits[:, :token_index_to_generate, :] = logits[:, :token_index_to_generate, :]         
                else:
                    all_logits[:, token_index_to_generate - 1, :] = logits[:, 0, :] # only one token will is computed

            # sample token id of the to be generated token
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
            
            
            yield context_tokens, token_generation_start_index, token_generation_end_index, all_logits, state_is_done.bool()
            if torch.all(state_is_done): break

def generate_samples_from_prompt(neox_args, model, text: Union[List[str], str], eos_token_id: int = None, 
                                    maximum_tokens: int = 64, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generates samples from raw text and returns them in a dictionary.

    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
    model: a Megatron model
    text: either a single prompt (str) or a list of prompts (List[str]).

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished': 
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds 
        
    """
    eos_token_id = eos_token_id or neox_args.tokenizer.eod

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

                if raw_text == "":
                    context_tokens = [eos_token_id]
                else:
                    context_tokens = neox_args.tokenizer.tokenize(raw_text)
                context_length = len(context_tokens)

                if context_length >= (neox_args.seq_length // 2):
                    print_rank_0("\nWarning! Context length", context_length,
                                 "\nPlease give smaller context (e.g. half of the "
                                 "max sequence length)!", flush=True)
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return generated_texts

        for batch_context_tokens, batch_token_generation_start_index, batch_token_generation_end_index, batch_logits, is_done in stream_tokens(
            neox_args=neox_args, 
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
            ):
            pass # finish generation and use all results below

        batch_context_tokens = batch_context_tokens.cpu().numpy().tolist()
        batch_token_generation_start_index = batch_token_generation_start_index.cpu().numpy().tolist()
        batch_token_generation_end_index = batch_token_generation_end_index.cpu().numpy().tolist()
        batch_is_done = is_done.cpu().numpy().tolist()
        for tokens, start_index, end_index, is_done in zip(batch_context_tokens, batch_token_generation_start_index, batch_token_generation_end_index, batch_is_done):
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
            if is_mp_rank_0():
                data = {
                    'context': raw_text, 
                    'text': generated_text, 
                    'length': len(generated_tokens), 
                    'finished': is_done, 
                    'message': message, 
                    'duration_seconds': float(time.time() - start_time)
                    }
                generated_texts.append(data)
                
    return generated_texts

def generate_samples_input_from_file(neox_args, model, input_file, output_file=None, eos_token_id: int = None, 
                                        maximum_tokens: int = 64, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generates samples from an input file and writes them to an output file.

    Reads prompts from neox_args.sample_input_file and writes completions to neox_args.sample_output_file

    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
    model: a Megatron model

    input_file: path to input file. Each line in the input file will be treated as separate prompt. The line break at the end of the line is not included in the prompt.
    output_file: file where generation results are to be stored in jsonl format. defaults to input_file+'.output.jsonl' if not defined

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0
    
    
    returns: List[dict] -> a list of dicts containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished': 
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds 
    """
    # Read the sample file
    print_rank_0('generate_samples_input_from_file() loading input from {}'.format(input_file))
    with open(input_file, "r") as f:
        prompts = f.readlines()
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if len(p) > 0]
    print_rank_0('generate_samples_input_from_file() prompts loaded: {}'.format(len(prompts)))
    
    if is_mp_rank_0():
        if output_file is None:
            output_file = str(input_file) + ".output.jsonl"
            print_rank_0('generate_samples_input_from_file() setting default output file to {}'.format(output_file))
        
    print_rank_0('generate_samples_input_from_file() generating...')
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args, 
        model=model, 
        text=prompts, 
        eos_token_id=eos_token_id, 
        maximum_tokens=maximum_tokens,
        recompute=recompute, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p
        )
    
    if is_mp_rank_0():
        with open(output_file, "w") as f_out:
            for item in generated_texts:
                f_out.write(json.dumps(item) + '\n')
    print_rank_0('generate_samples_input_from_file() done')
    return generated_texts

def generate_samples_unconditional(neox_args, model, number_of_samples: int = 10, output_file=None, eos_token_id: int = None, 
                                        maximum_tokens: int = 64, recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
    model: a Megatron model

    number_of_samples (default 10): number of unconditional samples to be generated
    
    output_file: file where generation results are to be stored in jsonl format. no file will be stored if ommitted

    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    maximum_tokens: maximum number of tokens to be generated

    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished': 
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds 
    """
   
    print_rank_0('generate_samples_unconditional() generating...')
    generated_texts = generate_samples_from_prompt(
        neox_args=neox_args, 
        model=model, 
        text=["" for _ in range(number_of_samples)], 
        eos_token_id=eos_token_id, 
        maximum_tokens=maximum_tokens, 
        recompute=recompute, 
        temperature=temperature, 
        top_k=top_k, 
        top_p=top_p
        )
    
    if is_mp_rank_0():
        if output_file is not None:
            with open(output_file, "w") as f_out:
                for item in generated_texts:
                    f_out.write(json.dumps(item) + '\n')
    print_rank_0('generate_samples_unconditional() done')
    return generated_texts

def generate_samples_interactive(neox_args, model, maximum_tokens: int = 64, eos_token_id: int = None, 
                                    recompute: bool = False, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0):
    """
    Generates samples unconditionially (no prompt) and yields them in a dictionary.

    neox_args: NeoXArgs with tokenizer, reset_position_ids, reset_attention_mask and eod_mask_loss
    model: a Megatron model

    maximum_tokens: maximum number of tokens to be generated
    eos_token_id: end of text token at which completion is terminated, even if max_tokes count has not been reached
    
    recompute: flag indicating whether a cache is used for already forwarded tokens (true) or whether all tokens are recomputed at every iteration (false)

    temperature (default 0.0): exponential scaling output distribution ("higher == more risk")
    top_k (default 0): integer -> integer between 0 and the models vocab size. Filters out any logits with a probability less than that of the top_kth token.
    top_p (default 0.0): float -> Top-p (nucleus) sampling chooses from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
    
    note: greedy decoding is used if temperature is 0.0, top_k is 0 and top_p is 0.0

    yields: dict containing the following fields:
        - 'context' (the input)
        - 'text' (the completion)
        - 'length' (the length of the completion in number of tokens)
        - 'finished': 
        - 'message': a messaged associated with the generation procedure, can be a warning or error
        - 'duration_seconds': duration of the generation in seconds 
    """
   
    while True:
        torch.distributed.barrier(group=mpu.get_model_parallel_group())
        terminate_runs = 0

        if torch.distributed.is_initialized() and  torch.distributed.get_rank() == 0:
            os.system('clear')
            raw_text = input("Context prompt >>> ")
            context_tokens = neox_args.tokenizer.tokenize(raw_text)
            if len(context_tokens) == 0:
                context_tokens = [neox_args.tokenizer.eod]
            context_length = len(context_tokens)
            if context_length >= (neox_args.seq_length - 1):
                print_rank_0("\nContext length"+str(context_length)+"\nReached max sequence length!")
                terminate_runs = 1
        else:
            context_tokens = neox_args.tokenizer.tokenize("EMPTY TEXT")
            context_length = len(context_tokens)

        
        terminate_runs = broadcast_terminate_signal(terminate_runs)
        if terminate_runs == 1:
            return
        for batch_context_tokens, batch_token_generation_start_index, batch_token_generation_end_index, batch_logits, is_done in stream_tokens(
            neox_args=neox_args, 
            model=model,
            context_tokens=[context_tokens],
            eos_token_id=eos_token_id,
            maximum_tokens=maximum_tokens,
            recompute=recompute,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
            ):
             if mpu.get_model_parallel_rank() == 0:
                generated_tokens = batch_context_tokens[0].cpu().numpy().tolist()[batch_token_generation_start_index[0].item():batch_token_generation_end_index[0].item()]
                generated_text = neox_args.tokenizer.detokenize(generated_tokens)
        
        print_rank_0("Generated Text: "+generated_text)
        if torch.distributed.is_initialized() and  torch.distributed.get_rank() == 0:
            _ = input("\n<press enter to continue>")


