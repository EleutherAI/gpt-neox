# coding=utf-8
#
# Copyright 2021 Biderman et al. This file is based on code by the authors denoted below and has been modified from its original version.
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

"""Utilities for models."""

import torch
from megatron.model.norms import LayerNorm, RMSNorm, ScaleNorm


def get_params_for_weight_decay_optimization(module, neox_args):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and biases will have no weight decay but the rest will.
    """
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module_ in module.modules():
        if any(
            [
                isinstance(module_, LayerNorm),
                isinstance(module_, RMSNorm),
                isinstance(module_, ScaleNorm),
            ]
        ) or (
            neox_args.weight_decay == 0.0
        ):  # also include all parameters here if no weight decay is being done
            no_weight_decay_params["params"].extend(
                [p for p in list(module_._parameters.values()) if p is not None]
            )
        else:
            weight_decay_params["params"].extend(
                [
                    p
                    for n, p in list(module_._parameters.items())
                    if p is not None and n != "bias"
                ]
            )
            no_weight_decay_params["params"].extend(
                [
                    p
                    for n, p in list(module_._parameters.items())
                    if p is not None and n == "bias"
                ]
            )
    if neox_args.weight_decay == 0.0:
        # only return a single param group
        # with onebitadam, we want to minimize the calls to compressed_allreduce. Every param group calls it once.
        # to avoid this, only use a single param group when weight decay is off.
        return [no_weight_decay_params]
    return weight_decay_params, no_weight_decay_params


def exists(x):
    return x is not None


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class SequentialWrapper(torch.nn.Module):
    """
    Used to convert a deepspeed PipelineModule to an nn.Sequential like model whilst retaining
    activation checkpointing.
    """

    def __init__(
        self,
        layers,
        activation_checkpoint_interval,
        activation_checkpoint_func,
        parent_class_name=None,
    ):
        super().__init__()
        self.sequential = torch.nn.Sequential(*layers)
        self.activation_checkpoint_interval = activation_checkpoint_interval
        self.parent_class_name = parent_class_name
        self.activation_checkpoint_func = activation_checkpoint_func

    def _is_checkpointable(self, funcs):
        if self.parent_class_name == "GPT2ModelPipe":
            return all(
                "ParallelTransformerLayerPipe" in f.__class__.__name__ for f in funcs
            )
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]
        return any(len(list(p)) > 0 for p in params)

    def inference_mode(self):
        _set_get_key_value(self.sequential, True)

    def train_mode(self):
        _set_get_key_value(self.sequential, False)

    def forward(self, forward_input):
        def exec_range_func(start, end):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.sequential[start:end]):
                    inputs = layer(inputs)
                return inputs

            return exec_func

        if self.activation_checkpoint_interval == 0:
            func = exec_range_func(0, len(self.sequential))
            x = func(forward_input)
        else:
            num_layers = len(self.sequential)
            x = forward_input
            for start_idx in range(0, num_layers, self.activation_checkpoint_interval):
                end_idx = min(
                    start_idx + self.activation_checkpoint_interval, num_layers
                )

                funcs = self.sequential[start_idx:end_idx]
                # Since we either pass tensors or tuples of tensors without unpacking, we
                # need to be careful not to double-wrap tensors with tuple.
                if not isinstance(x, tuple):
                    x = (x,)

                if self._is_checkpointable(funcs):
                    x = self.activation_checkpoint_func(
                        exec_range_func(start_idx, end_idx), *x
                    )
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x


def _set_get_key_value(base, value):
    # utility used to recursively set the get key attribute value to true/false
    # (i.e switch between inference and train mode)

    assert isinstance(value, bool)
    for m in base:
        if hasattr(m, "get_key_value"):
            m.get_key_value = value
        if hasattr(m, "children"):
            _set_get_key_value(m.children(), value)


def configure_sparse_attention(neox_args, attention_type, num_attention_heads, mpu):
    from deepspeed.ops.sparse_attention import (
        SparseSelfAttention,
        VariableSparsityConfig,
        FixedSparsityConfig,
        BigBirdSparsityConfig,
        BSLongformerSparsityConfig,
    )
    from deepspeed.ops.sparse_attention.sparsity_config import (
        LocalSlidingWindowSparsityConfig,
    )

    if attention_type == "sparse_fixed":
        # you can think of local window size as `block_size` * `num_local_blocks`.
        # so if you wanted to set a local window size of 256, set block size to 16 and `num_local_blocks` to 16
        sparsity_config = FixedSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_local_blocks=neox_args.sparsity_config.get("num_local_blocks", 4),
            num_global_blocks=neox_args.sparsity_config.get("num_global_blocks", 1),
            num_different_global_patterns=neox_args.sparsity_config.get(
                "num_different_global_patterns", 1
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "sparse_variable":
        sparsity_config = VariableSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=neox_args.sparsity_config.get("num_random_blocks", 0),
            local_window_blocks=neox_args.sparsity_config.get(
                "local_window_blocks", [4]
            ),
            global_block_indices=neox_args.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=neox_args.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
            horizontal_global_attention=False,
        )
    elif attention_type == "local":
        # can configure with `num_local_blocks` or `num_sliding_window_blocks`
        num_local_blocks = neox_args.sparsity_config.get(
            "num_local_blocks",
            neox_args.sparsity_config.get("num_sliding_window_blocks", 4),
        )
        sparsity_config = LocalSlidingWindowSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            num_sliding_window_blocks=num_local_blocks,
            attention="unidirectional",
        )
    elif attention_type == "bigbird":
        sparsity_config = BigBirdSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_random_blocks=neox_args.sparsity_config.get("num_random_blocks", 1),
            num_sliding_window_blocks=neox_args.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            num_global_blocks=neox_args.sparsity_config.get("num_global_blocks", 1),
            attention="unidirectional",
        )
    elif attention_type == "bslongformer":
        sparsity_config = BSLongformerSparsityConfig(
            num_heads=num_attention_heads,
            block=neox_args.sparsity_config.get("block", 16),
            different_layout_per_head=neox_args.sparsity_config.get(
                "different_layout_per_head", False
            ),
            num_sliding_window_blocks=neox_args.sparsity_config.get(
                "num_sliding_window_blocks", 3
            ),
            global_block_indices=neox_args.sparsity_config.get(
                "global_block_indices", [0]
            ),
            global_block_end_indices=neox_args.sparsity_config.get(
                "global_block_end_indices", None
            ),
            attention="unidirectional",
        )
    else:
        raise ValueError(f"Attention type {attention_type} not recognized")
    return SparseSelfAttention(
        sparsity_config=sparsity_config,
        max_seq_length=neox_args.seq_length,
        attn_mask_mode="add",
        mpu=mpu,
    )
