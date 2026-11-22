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

from contextlib import nullcontext

import pytest
import torch

from megatron import mpu
from megatron.model.gpt2_model import gpt2_attention_mask_func
from megatron.model.transformer import ParallelTransformerLayer
from megatron.neox_arguments import NeoXArgs

pytestmark = pytest.mark.cpu


def _make_block(monkeypatch, hidden_dropout):
    monkeypatch.setattr(mpu.initialize, "_MPU_WORLD_SIZE", 1)
    monkeypatch.setattr(mpu.initialize, "_MPU_RANK", 0)
    # Attention dropout is disabled, so the CPU test needs no CUDA RNG stream.
    monkeypatch.setattr(mpu.get_cuda_rng_tracker(), "fork", nullcontext)
    args = NeoXArgs(
        global_num_gpus=1,
        train_micro_batch_size_per_gpu=1,
        use_cpu_initialization=True,
        num_layers=1,
        hidden_size=32,
        intermediate_size=96,
        num_attention_heads=4,
        seq_length=4,
        max_position_embeddings=4,
        activation="swiglu",
        gpt_j_residual=True,
        use_bias_in_mlp=False,
        hidden_dropout=hidden_dropout,
        attention_dropout=0.0,
    )
    layer = ParallelTransformerLayer(
        args,
        attention_mask_func=gpt2_attention_mask_func,
        init_method=torch.nn.init.xavier_uniform_,
        output_layer_init_method=torch.nn.init.xavier_uniform_,
        layer_number=0,
    )
    x = torch.randn(args.seq_length, 1, args.hidden_size)
    mask = torch.ones(args.seq_length, args.seq_length, dtype=torch.bool).triu(1)
    return layer, x, mask


@torch.no_grad()
def test_gpt_j_residual_preserves_attention_with_zero_mlp(monkeypatch):
    layer, x, mask = _make_block(monkeypatch, hidden_dropout=0.0)
    layer.eval()
    layer.mlp.linear2.weight.zero_()

    attention, bias = layer.attention(layer.input_layernorm(x), mask)
    attention = attention + bias
    assert attention.count_nonzero() > 0
    torch.testing.assert_close(layer(x, mask), x + attention)


@torch.no_grad()
def test_gpt_j_residual_applies_dropout_to_bias_free_mlp(monkeypatch):
    layer, x, mask = _make_block(monkeypatch, hidden_dropout=1.0)
    layer.attention.dense.weight.zero_()
    layer.attention.dense.bias.zero_()
    mlp_output, bias = layer.mlp(layer.post_attention_layernorm(x))
    assert bias is None
    assert mlp_output.count_nonzero() > 0

    layer.train()
    torch.testing.assert_close(layer(x, mask), x, rtol=0, atol=0)
    layer.eval()
    torch.testing.assert_close(layer(x, mask), x + mlp_output)
