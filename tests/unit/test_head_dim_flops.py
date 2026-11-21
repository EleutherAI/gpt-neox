# Copyright (c) 2026, EleutherAI
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

from types import SimpleNamespace

import pytest
import torch

from megatron import logging as megatron_logging


def flops_args(**overrides):
    values = {
        "padded_vocab_size": 50304,
        "train_batch_size": 8,
        "seq_length": 2048,
        "hidden_size": 1024,
        "num_layers": 28,
        "num_attention_heads": 16,
        "num_kv_heads": None,
        "head_dim": None,
        "checkpoint_activations": False,
        "attention_config": ["global"] * 28,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def legacy_square_attention_flops(args, pass_factor):
    hidden_size = args.hidden_size
    seq_len = args.seq_length
    num_layers = args.num_layers
    return (
        24
        * pass_factor
        * args.train_batch_size
        * seq_len
        * num_layers
        * (hidden_size**2)
        * (
            1.0
            + (seq_len / (6.0 * hidden_size))
            + (args.padded_vocab_size / (16.0 * num_layers * hidden_size))
        )
    )


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)


@pytest.mark.cpu
@pytest.mark.parametrize("head_dim", [None, 64])
@pytest.mark.parametrize("checkpoint_activations", [False, True])
def test_square_attention_flops_are_unchanged(head_dim, checkpoint_activations):
    args = flops_args(head_dim=head_dim, checkpoint_activations=checkpoint_activations)
    actual_factor = 4 if checkpoint_activations else 3

    assert megatron_logging.get_actual_flops(args, 1.0) == (
        legacy_square_attention_flops(args, actual_factor)
    )
    assert megatron_logging.get_forward_backward_flops(args, 1.0) == (
        legacy_square_attention_flops(args, 3)
    )


@pytest.mark.cpu
def test_flops_count_explicit_head_dim_and_gqa_widths():
    # Qwen3-0.6B attention: 16 query heads and 8 KV heads of width 128.
    args = flops_args(num_kv_heads=8, head_dim=128)
    hidden_size = args.hidden_size
    seq_len = args.seq_length
    query_width, kv_width = 2048, 1024
    forward_flops_per_token_per_layer = (
        2 * hidden_size * (query_width + 2 * kv_width)  # fused QKV projection
        + 2 * query_width * hidden_size  # output projection
        + 4 * seq_len * query_width  # attention scores and context
        + 16 * hidden_size**2  # MLP
    )
    expected = (
        3
        * args.train_batch_size
        * seq_len
        * (
            args.num_layers * forward_flops_per_token_per_layer
            + 1.5 * hidden_size * args.padded_vocab_size
        )
    )

    assert megatron_logging.get_forward_backward_flops(args, 1.0) == pytest.approx(
        expected, rel=1e-12
    )
    assert megatron_logging.get_forward_backward_flops(args, 1.0) != pytest.approx(
        legacy_square_attention_flops(args, 3), rel=1e-3
    )
