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
from types import SimpleNamespace

import pytest
import torch

from megatron import mpu
from megatron.model.transformer import ParallelTransformerLayer
from megatron.neox_arguments import NeoXArgs


@pytest.mark.cpu
@pytest.mark.parametrize("norm_placement", ["pre_norm", "post_norm_inside_residual"])
def test_transformer_norm_placement(monkeypatch, norm_placement):
    monkeypatch.setattr(mpu.initialize, "_MPU_WORLD_SIZE", 1)
    monkeypatch.setattr(mpu.initialize, "_MPU_RANK", 0)
    monkeypatch.setattr(
        mpu, "get_cuda_rng_tracker", lambda: SimpleNamespace(fork=nullcontext)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))

    args = NeoXArgs.from_ymls(
        ["tests/config/test_setup.yml", "configs/cpu_mock_config.yml"],
        overwrite_values={
            "use_cpu_initialization": True,
            "precision": "fp32",
            "norm": "rmsnorm",
            "norm_placement": norm_placement,
            "use_bias_in_attn_linear": False,
            "use_bias_in_mlp": False,
            "activation": "swiglu",
            "intermediate_size": 36,
            "use_qk_norm": True,
            "qk_norm_type": "across_heads",
            "qk_norm_separate": True,
            "qk_norm_across_tp": True,
            "num_kv_heads": 2,
        },
    )
    layer = ParallelTransformerLayer(
        args, None, torch.nn.init.xavier_uniform_, torch.nn.init.xavier_uniform_, 0
    )
    x = torch.randn(3, 2, args.hidden_size)

    if norm_placement == "pre_norm":
        attention, _ = layer.attention(layer.input_layernorm(x), None)
        residual = x + attention
        mlp, _ = layer.mlp(layer.post_attention_layernorm(residual))
        expected = residual + mlp
    else:
        attention, _ = layer.attention(x, None)
        residual = x + layer.post_attention_layernorm(attention)
        mlp, _ = layer.mlp(residual)
        expected = residual + layer.post_feedforward_layernorm(mlp)

    assert torch.allclose(layer(x, None), expected)
