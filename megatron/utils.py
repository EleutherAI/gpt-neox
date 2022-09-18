# Copyright (c) 2021 Josh Levy-Kramer <josh@levykramer.co.uk>.
# This file is based on code by the authors denoted below and has been modified from its original version.
#
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

"""General utilities."""
import os
import sys
import re
import time
import socket
from typing import Dict, List

import requests
import wandb
from wandb import UsageError

import torch

from deepspeed.launcher.runner import fetch_hostfile, parse_inclusion_exclusion

from megatron import print_rank_0
from megatron import mpu
from deepspeed import PipelineEngine, DeepSpeedEngine
from collections import deque


def reduce_losses(losses):
    """Reduce a tensor of losses across all GPUs."""
    reduced_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(reduced_losses)
    reduced_losses = reduced_losses / torch.distributed.get_world_size()
    return reduced_losses


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(
        torch.cuda.max_memory_allocated() / mega_bytes
    )
    string += " | reserved: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max reserved: {}".format(
        torch.cuda.max_memory_reserved() / mega_bytes
    )
    print_rank_0(string)


def get_full_mask(src_length, target_length, device):
    """
    Get a full (non-triangular, all tokens attending eachother) potentially non-square mask
    """
    # TODO(Hailey): change naming convention in this function (swap src and "target") I think
    mask = torch.ones((1, src_length, target_length), device=device).view(
        1, 1, src_length, target_length
    )

    # convert to binary
    return mask < 0.5


def _get_attn_mask(
    seq_length, 
    device, 
    prefix_indices=None, 
    decoder_is_inputs=None,
    segment_ids=None,
    batch_size=1, 
    neox_args=None,
    ):
    """
    Get attention mask for a given batch and device.
    """

    mask = _get_causal_attn_mask(seq_length, device, batch_size=batch_size)
    
    # Prefix lm per row, if using prefixlm or mlm and NOT mtf (no packing)
    if prefix_indices is not None:
        for b in range(batch_size):
            # prefix_indices[b] is an int here
            mask[b, 0, :prefix_indices[b], :prefix_indices[b]] = 1

    # convert to bool
    return mask < 0.5


def _get_packed_masks_and_position_ids(
    data,
    causal_mask,
    loss_mask,
    position_ids,
    decoder_is_inputs,
    segment_ids,
    neox_args=None,
    ):
    """
    Function based on a similar function from Bigscience Meg-DS fork:
    https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/0f23a729ec5ed6ef46e1fecc8071ac11acdb91c6/megatron/utils.py#L253

    Which was inspired by 
    https://github.com/google-research/t5x/blob/7193407f98a8b18100b71a04ff777238be1682ca/t5x/examples/decoder_only/layers.py#L978

    This function takes in 

    Arguments:
        - causal_mask: torch.BoolTensor [batch_size, sequence_length, sequence_length]
        - decoder_is_inputs: torch.BoolTensor [batch_size, sequence_length]
        - segment_ids: torch.IntTensor [batch_size, sequence_length]
    Returns:
        - attention_mask: torch.BoolTensor [batch_size, 1, sequence_length, sequence_length]
        - loss_mask: 
    """
    batch_size = causal_mask.size()[0]
    eod_token = neox_args.tokenizer.eod

    # using packing (w/ multi-task finetuning), so need to reset position ids for each segment to start at 0.
    for b in range(batch_size):

        # locations of EOD tokens, denoting breaks between segments.
        eod_idxs = position_ids[b, data[b] == eod_token]

        # TODO(Hailey): check that this comment below still applies
        # If the last eod token is not the last token of the sequence, we suppose that there is a partial document
        # We treat this case as if we add an eod token at the end of the sequence.
        if data[b][-1] != eod_token:
            eod_idxs= torch.cat(
                (eod_idxs, torch.tensor([len(data[b])], dtype=eod_idxs.dtype, device=eod_idxs.device))
            )

        # (TODO(Hailey): this is from Meg-DS, but is it needed?)
        # decouple EOD locations from position ids 
        eod_idxs = eod_idxs.detach().clone()

        # Loop through all EOD locations, resetting position ids of each segment to start @ 0.
        prev_segment_start_idx = 0 
        for j in range(eod_idxs.size()[0]):
            # i = j-th location of an EOD token
            i = eod_idxs[j]

            # Prevent cross document attention interactions.
            causal_mask[b, 0, (i + 1):, :(i + 1)] = 0

            # TODO(Hailey): this commented codeblock needs reworking in order to use prefixlm w/ MTF.
            # we probably can do prefixlm using decoder_is_inputs, instead of passing prefix_indices.

            # # Prefix lm per document.
            # if prefix_indices:
            #     assert isinstance(prefix_indices[b], list), f"prefix for a row has to be document specific, and consequently return a list, got {prefix_indices[b]}"
            #     attention_mask[b, 0, prev_index: prefix_indices[b][j], prev_index: prefix_indices[b][j]] = 1
            #     if loss_on_targets_only:
            #         # Last token of the prefix should predict the prefix_index id
            #         loss_mask[b, prev_index: prefix_indices[b][j] - 1] = 0.0

            # Reset position ids to start from 0.
            position_ids[b, (i + 1):] -= (i + 1 - prev_segment_start_idx)

            # remember the place that the next segment starts.
            prev_segment_start_idx = i + 1

    # loss masking: only compute loss signal on target tokens.
    if neox_args.loss_on_targets_only:
        # TODO(Hailey): are we sure we want to shift inputs via the [:, 1:] here?
        is_target = ~decoder_is_inputs[:, 1:]

        # EOD token loss was already masked, so just mask these now.
        loss_mask *= is_target

    # TODO(Hailey): look at t5x impl. to see how to make this shorter; remove asserts
    """Causal Inputs Mask:
    mask = [[[[1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1]]]]
    """
    assert causal_mask.dtype == torch.bool
    assert segment_ids.dtype == torch.long
    # Make attn bidirectional over inputs. but only if we are using prefixLM (non-causal decoder).
    if neox_args.training_objective != "prefixlm":
        causal_inputs_mask = causal_mask
    else:
        # shift decoder_is_inputs labels at this step.
        decoder_is_inputs = decoder_is_inputs[:, :-1].bool()
        inputs_mask = decoder_is_inputs[:, None, :, None] * decoder_is_inputs[:, None, None, :]
        causal_inputs_mask = causal_mask + inputs_mask

    """Padding Mask:
    mask = [[[[1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    padding_mask = (segment_ids != 0)[:, None, :, None] * (segment_ids != 0)[:, None, None, :]

    """Segment Mask:
    mask = [[[[1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    segment_mask = segment_ids[:, None, :, None] == segment_ids[:, None, None, :]

    """Final Mask:
    mask = [[[[1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0]]]]
    """
    attention_mask = causal_inputs_mask * padding_mask * segment_mask

    return attention_mask, loss_mask, position_ids


def _get_causal_attn_mask(
    seq_length, 
    device, 
    batch_size=1,
    ):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((batch_size, seq_length, seq_length), device=device)).view(
        batch_size, 1, seq_length, seq_length
    )

    return mask


def _get_loss_mask(data, prefix_indices=None, neox_args=None):
    """
    Get loss mask for a input sequence. Accounts for masking EOD token loss and 
    getting loss signal over target tokens only (prefixlm/mlm objectives).
    """
    # extract args from neox_args
    eod_token = neox_args.tokenizer.eod
    eod_mask_loss = neox_args.eod_mask_loss
    loss_on_targets_only = neox_args.loss_on_targets_only

    # create loss mask
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # mask EOD tokens if desired
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # mask loss calc over the inputs, if using prefixlm or mlm (not packed)
    if loss_on_targets_only and not neox_args.train_mtf:
        if prefix_indices is not None:
            for b in range(data.size()[0]):
                # Last token of the prefix should predict the prefix_index id
                loss_mask[b, :prefix_indices[b] - 1] = 0.0

    return loss_mask


def get_ltor_masks_and_position_ids(
    data,
    prefix_indices=None,
    decoder_is_inputs=None,
    segment_ids=None,
    neox_args=None,
):
    """
    Build masks and position ids.
    `prefix_indices` can have multiple types:
        - None signifies that the model is fully autoregressive. ("clm" objective)
        - List[int] the argument holds all prefix indices that split a row into an input and a target

    """

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (not necessarily lower triangular).
    attention_mask = _get_attn_mask(
        seq_length=seq_length,
        device=data.device,
        prefix_indices=prefix_indices,
        decoder_is_inputs=decoder_is_inputs,
        segment_ids=segment_ids,
        batch_size=batch_size,
        neox_args=neox_args,
    )
    # TODO(Hailey:) a final check that loss masking is done right (incl. w/ padding + mask toks)
    # Loss mask.
    loss_mask = _get_loss_mask(data, prefix_indices=prefix_indices, neox_args=neox_args)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    # if packing is done, need extra processing of position ids + attn mask + loss mask
    if neox_args.train_mtf:
        attention_mask, loss_mask, position_ids = _get_packed_masks_and_position_ids(
            data,
            attention_mask,
            loss_mask,
            position_ids,
            decoder_is_inputs,
            segment_ids,
            neox_args=neox_args,
        )

    return attention_mask, loss_mask, position_ids


def get_position_ids(data):
    """Create position ids for a given batch."""

    batch_size, seq_length = data.size()

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)

    return position_ids


def local_rank():
    """Local rank of process"""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        print(
            "utils.local_rank() environment variable LOCAL_RANK not set, defaulting to 0",
            flush=True,
        )
        local_rank = 0
    return int(local_rank)


def is_bnb_available():
    """True if bitsandbytes optimizers are available"""
    return importlib.util.find_spec("bitsandbytes") is not None


def is_local_main():
    """True if is the local main process"""
    return local_rank() == 0


def is_mp_rank_0():
    """True if mp rank == 0"""
    return mpu.get_model_parallel_rank() == 0


def get_wandb_api_key(neox_args):
    """Get Weights and Biases API key from ENV or .netrc file. Otherwise return None"""
    if "WANDB_LOCAL" in os.environ:
        return "LOCAL"
    if "WANDB_API_KEY" in os.environ:
        return os.environ["WANDB_API_KEY"]

    wandb_token = requests.utils.get_netrc_auth(neox_args.wandb_host)

    if wandb_token is not None:
        return wandb_token[1]


def init_wandb(neox_args):
    # Wandb. (one worker per machine)
    if neox_args.use_wandb == False:
        return

    if not neox_args.wandb_init_all_ranks:
        use_wandb = is_local_main() and (
            get_wandb_api_key(neox_args=neox_args) is not None
        )
        neox_args.update_value("use_wandb", use_wandb)
    if neox_args.use_wandb:
        group_name = neox_args.wandb_group
        name = f"{socket.gethostname()}-{local_rank()}" if group_name else None
        try:
            wandb.init(
                project=neox_args.wandb_project,
                group=group_name,
                name=name,
                save_code=False,
                force=False,
                entity=neox_args.wandb_team,
            )
        except UsageError as e:
            neox_args.update_value("use_wandb", False)
            print(e)
            print(
                "Skipping wandb. Execute `wandb login` on local or main node machine to enable.",
                flush=True,
            )
        wandb.config.update(neox_args.all_config)


def obtain_resource_pool(
    hostfile_path, include_arg, exclude_arg
) -> Dict[str, List[int]]:
    """
    Get dict of `resource_pool[hostname] = [list of GPU ranks]` using hostfile, include and exclude args.
    Modified from: `deepspeed.launcher.runner.main`
    """
    resource_pool = fetch_hostfile(hostfile_path)
    if not resource_pool:
        resource_pool = {}
        device_count = torch.cuda.device_count()
        if device_count == 0:
            raise RuntimeError("Unable to proceed, no GPU resources available")
        resource_pool["localhost"] = device_count

    active_resources = parse_inclusion_exclusion(
        resource_pool, include_arg, exclude_arg
    )
    return active_resources


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def ddb(rank=0):
    """
    Distributed Debugger that will insert a py debugger on rank `rank` and
    pause all other distributed processes until debugging is complete.
    :param rank:
    """
    if torch.distributed.get_rank() == rank:
        from pdb import Pdb

        pdb = Pdb(skip=["torch.distributed.*"])
        pdb.set_trace(sys._getframe().f_back)
    torch.distributed.barrier()


class Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()

    def start(self):
        """Start the timer."""
        assert not self.started_, "timer has already been started"
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        torch.cuda.synchronize()
        self.elapsed_ += time.time() - self.start_time
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self, use_wandb, tensorboard_writer):
        self.timers = {}
        self.use_wandb = use_wandb
        self.tensorboard_writer = tensorboard_writer

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    def write(self, names, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # pollutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer

            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar(f"timers/{name}", value, iteration)

            if self.use_wandb:
                wandb.log({f"timers/{name}": value}, step=iteration)

    def log(self, names, normalizer=1.0, reset=True):
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                print(string, flush=True)
        else:
            print(string, flush=True)


def expand_attention_types(attention_config, num_layers):
    """
    Expands an `attention_config` list in the following format:

        [
        [['attention_type_1', ..., `attention_type_n`], 12]
        ]

    to a flattened list of length `num_layers`.

    :param params_list:
    :return:
    """
    # if only strings are found in the config, we assume it's already expanded
    if all([isinstance(i, str) for i in attention_config]):
        return attention_config
    newlist = []
    for item in attention_config:
        # instead of specifying a number - we can specify 'all' to extend this pattern across all layers
        if item[1] == "all":
            assert num_layers % len(item[0]) == 0, (
                f"Number of layers ({num_layers}) is not divisible by the length "
                f"of pattern: {item[0]}"
            )
            return item[0] * (num_layers // len(item[0]))
        for _ in range(item[1]):
            newlist.extend(item[0])
    return newlist


class OverflowMonitor:

    """
    Checks if the past n iterations have been skipped due to overflow, and exits
    training if that happens.
    """

    def __init__(self, optimizer, n=50):
        self.optimizer = optimizer
        self.n = n
        self.history = deque(maxlen=n)

    def check(self, skipped):
        self.history.append(skipped)
        if (
            self.optimizer.overflow
            and len(self.history) == self.n
            and all(self.history)
        ):
            raise Exception(
                f"Skipped {self.n} iterations in a row due to Overflow - Exiting training."
            )


def get_noise_scale_logger(neox_args):
    if neox_args.log_gradient_noise_scale:
        if neox_args.zero_stage >= 1:
            raise NotImplementedError(
                "Gradient Noise Scale logging does not work with zero stage 2+, as the "
                "gradients are distributed across ranks."
            )
        noise_scale_logger = GradientNoiseScale(
            model=model,
            batch_size_small=neox_args.train_batch_size,
            n_batches=neox_args.gradient_noise_scale_n_batches,
            cpu_offload=neox_args.gradient_noise_scale_cpu_offload,
            neox_args=neox_args,
            mpu=mpu,
        )
    else:
        noise_scale_logger = None
    return noise_scale_logger


def get_total_params(model):
    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        params = sum([p.nelement() for p in model.parameters()])
        print(
            " > number of parameters on model parallel rank {}: {}".format(
                mpu.get_model_parallel_rank(), params
            ),
            flush=True,
        )
    else:
        params = 0

    total_n_parameters = torch.tensor([params]).cuda(torch.cuda.current_device())
    torch.distributed.all_reduce(total_n_parameters)
    total_n_parameters = total_n_parameters.item()
    return total_n_parameters


def setup_for_inference_or_eval(
    use_cache=True,
    overwrite_values=None,
):
    """
    Initializes the model for evaluation or inference (doesn't load optimizer states, etc.) from command line args.

    use_cache: bool
        Whether to use key value caching in inference.
    overwrite_values: dict
        Optional Values to overwrite in the model config.
    """

    from megatron.neox_arguments import NeoXArgs
    from megatron.initialize import initialize_megatron
    from megatron.training import setup_model_and_optimizer

    _overwrite_values = {
        "checkpoint_activations": False,
        "partition_activations": False,
        "no_load_optim": True,
        "zero_optimization": None,  # disable zero optimization (won't be used in inference, and loading zero optimizer can cause errors)
    }
    if overwrite_values:
        _overwrite_values.update(overwrite_values)
    neox_args = NeoXArgs.consume_neox_args(overwrite_values=_overwrite_values)
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()

    if neox_args.load is None:
        raise ValueError("`load` parameter must be supplied to load a model`")

    # initialize megatron
    initialize_megatron(neox_args)

    # set up model and load checkpoint.
    model, _, _ = setup_model_and_optimizer(
        neox_args=neox_args,
        use_cache=use_cache,
        iteration=neox_args.iteration,
    )  # we use setup_model_and_optimizer instead of get_model in order to initialize deepspeed
    print_rank_0("Finished loading model")

    model.module.inference_mode(use_cache=use_cache)
    return model, neox_args


class CharCounter:
    """
    Wraps the data_iterator to count the number of characters in a batch
    """

    def __init__(self, data_iterator, tokenizer):
        self.tokenizer = tokenizer
        self.data_iterator = data_iterator
        self.char_count = 0
        self.batch_count = 0
        self.token_count = 0
        self.total_time = 0

    def tokens_per_char(self):
        return self.token_count / self.char_count

    def __iter__(self):
        return self

    def __next__(self):
        start = time.time()
        batch = self.data_iterator.__next__()
        for b in batch["text"]:
            self.token_count += len(b)
            self.char_count += len(self.tokenizer.detokenize(b.tolist()))
        self.batch_count += 1
        end = time.time()
        self.total_time += end - start
        return batch
