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

from contextlib import contextmanager, nullcontext
from copy import deepcopy
import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import megatron.neox_arguments.arguments as neox_arguments
from megatron import mpu
from megatron.mpu import layers as mpu_layers
from megatron.model.init_functions import init_method_normal
from megatron.model.transformer import ParallelSelfAttention
from megatron.model.utils import get_attention_head_dim
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(
        attention_mask, torch.finfo(attention_scores.dtype).min
    )


def attention_args(
    *,
    hidden_size,
    num_attention_heads,
    head_dim,
    num_kv_heads=None,
    model_parallel_size=1,
    **overrides,
):
    config = deepcopy(BASE_CONFIG)
    config.update(
        {
            "global_num_gpus": model_parallel_size,
            "model_parallel_size": model_parallel_size,
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "pos_emb": "none",
            "precision": "fp32",
            "use_cpu_initialization": True,
        }
    )
    config.update(overrides)
    return NeoXArgs.from_dict(config)


@contextmanager
def model_parallel_context(world_size):
    mpu.destroy_model_parallel()
    mpu.set_model_parallel_world_size(world_size)
    mpu.set_model_parallel_rank(0)
    try:
        yield
    finally:
        mpu.destroy_model_parallel()


def build_attention(neox_args, *, rotary=False, use_cache=False):
    return ParallelSelfAttention(
        neox_args=neox_args,
        attention_mask_func=attention_mask_func,
        init_method=init_method_normal(0.02),
        output_layer_init_method=init_method_normal(0.02),
        layer_number=0,
        rotary=rotary,
        use_cache=use_cache,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("model_parallel_size", [1, 2])
def test_qwen_style_gqa_projection_shapes(model_parallel_size):
    neox_args = attention_args(
        hidden_size=1024,
        num_attention_heads=16,
        num_kv_heads=8,
        head_dim=128,
        model_parallel_size=model_parallel_size,
    )

    with model_parallel_context(model_parallel_size):
        attention = build_attention(neox_args)

    assert attention.hidden_size_per_attention_head == 128
    assert attention.query_hidden_size == 2048
    assert attention.query_hidden_size_per_partition == 2048 // model_parallel_size
    assert attention.kv_hidden_size == 1024
    assert attention.query_key_value.output_size == 4096
    assert attention.query_key_value.weight.shape == (
        4096 // model_parallel_size,
        1024,
    )
    assert attention.dense.input_size == 2048
    assert attention.dense.output_size == 1024
    assert attention.dense.weight.shape == (
        1024,
        2048 // model_parallel_size,
    )
    assert attention.norm_factor == math.sqrt(128)


@pytest.mark.cpu
@pytest.mark.parametrize(
    (
        "hidden_size",
        "num_attention_heads",
        "num_kv_heads",
        "expected_query_hidden_size",
        "expected_kv_hidden_size",
        "expected_qkv_hidden_size",
    ),
    [
        (2560, 32, 8, 4096, 1024, 6144),
        (5120, 64, 8, 8192, 1024, 10240),
    ],
)
def test_large_qwen_attention_geometry_without_parameter_allocation(
    hidden_size,
    num_attention_heads,
    num_kv_heads,
    expected_query_hidden_size,
    expected_kv_hidden_size,
    expected_qkv_hidden_size,
):
    neox_args = attention_args(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        model_parallel_size=2,
    )

    head_dim = get_attention_head_dim(neox_args)
    query_hidden_size = neox_args.num_attention_heads * head_dim
    kv_hidden_size = neox_args.num_kv_heads * head_dim

    assert query_hidden_size == expected_query_hidden_size
    assert kv_hidden_size == expected_kv_hidden_size
    assert query_hidden_size + 2 * kv_hidden_size == expected_qkv_hidden_size


@pytest.mark.cpu
@pytest.mark.parametrize("head_dim", [None, 4])
def test_legacy_gqa_projection_shapes_are_unchanged(head_dim):
    neox_args = attention_args(
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
    )

    with model_parallel_context(1):
        attention = build_attention(neox_args)

    assert get_attention_head_dim(neox_args) == 4
    assert attention.query_hidden_size == 16
    assert attention.kv_hidden_size == 8
    assert attention.query_key_value.weight.shape == (32, 16)
    assert attention.dense.weight.shape == (16, 16)


@pytest.mark.cpu
def test_legacy_attention_state_dict_loads_with_derived_head_dim():
    legacy_args = attention_args(
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=None,
    )
    explicit_args = attention_args(
        hidden_size=16,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=4,
    )

    with model_parallel_context(1):
        legacy_attention = build_attention(legacy_args)
        explicit_attention = build_attention(explicit_args)
        explicit_attention.load_state_dict(legacy_attention.state_dict(), strict=True)

    assert (
        legacy_attention.state_dict().keys() == explicit_attention.state_dict().keys()
    )


@pytest.mark.cpu
def test_mha_projection_can_expand_beyond_residual_width():
    neox_args = attention_args(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=6,
    )

    with model_parallel_context(1):
        attention = build_attention(neox_args)

    assert not attention.gqa
    assert attention.query_hidden_size == 12
    assert attention.kv_hidden_size == 12
    assert attention.query_key_value.weight.shape == (36, 8)
    assert attention.dense.weight.shape == (8, 12)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("rotary_pct", "expected_rotary_ndims", "expected_embedding_dim"),
    [(0.5, 4, 4), (1.0, None, 8)],
)
def test_rotary_uses_explicit_head_dim(
    rotary_pct, expected_rotary_ndims, expected_embedding_dim
):
    neox_args = attention_args(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=8,
        pos_emb="rotary",
        rotary_pct=rotary_pct,
    )

    with model_parallel_context(1):
        attention = build_attention(neox_args, rotary=True)

    assert attention.rotary_ndims == expected_rotary_ndims
    assert attention.rotary_emb.dim == expected_embedding_dim


@pytest.mark.cpu
def test_qk_layernorm_shape_uses_explicit_head_dim():
    neox_args = attention_args(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=8,
        use_qk_layernorm=True,
    )

    with model_parallel_context(1):
        attention = build_attention(neox_args)

    assert attention.qk_layernorm.normalized_shape == (2, 8)


@pytest.mark.cpu
@pytest.mark.parametrize(
    (
        "num_attention_heads",
        "num_kv_heads",
        "head_dim",
        "expected_qkv_width",
    ),
    [(2, None, 6, 36), (4, 2, 4, 32)],
)
@pytest.mark.parametrize("model_parallel_size", [1, 2])
def test_attention_forward_backward_and_cache_use_independent_widths(
    monkeypatch,
    num_attention_heads,
    num_kv_heads,
    head_dim,
    expected_qkv_width,
    model_parallel_size,
):
    neox_args = attention_args(
        hidden_size=8,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        model_parallel_size=model_parallel_size,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        mpu,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(fork=nullcontext),
    )
    # Exercise one local TP shard without initializing torch.distributed. The
    # collective wrappers do not change tensor geometry, which is what this
    # focused CPU test verifies.
    monkeypatch.setattr(mpu_layers, "copy_to_model_parallel_region", lambda x: x)
    monkeypatch.setattr(mpu_layers, "reduce_from_model_parallel_region", lambda x: x)

    with model_parallel_context(model_parallel_size):
        attention = build_attention(neox_args, use_cache=True)
        hidden_states = torch.randn(3, 2, 8, requires_grad=True)
        attention_mask = torch.triu(
            torch.ones(1, 1, 3, 3, dtype=torch.bool), diagonal=1
        )

        (output, present), _ = attention(hidden_states, attention_mask)
        decode_states = torch.randn(1, 2, 8, requires_grad=True)
        decode_mask = torch.zeros(1, 1, 1, 4, dtype=torch.bool)
        (decode_output, decode_present), _ = attention(
            decode_states,
            decode_mask,
            layer_past=present,
        )
        (output.square().mean() + decode_output.square().mean()).backward()

    assert output.shape == (3, 2, 8)
    assert decode_output.shape == (1, 2, 8)
    expected_cache_heads = num_attention_heads // model_parallel_size
    assert present.shape == (2, 3, 2, expected_cache_heads, head_dim)
    assert decode_present.shape == (2, 4, 2, expected_cache_heads, head_dim)
    assert attention.query_key_value.weight.shape == (
        expected_qkv_width // model_parallel_size,
        8,
    )
    assert attention.query_key_value.weight.grad is not None
    assert attention.dense.weight.grad is not None
    assert hidden_states.grad is not None
    assert decode_states.grad is not None


@pytest.mark.cpu
def test_flash_gqa_projection_retains_kv_head_count(monkeypatch):
    monkeypatch.setattr(neox_arguments, "version", lambda _package: "2.5.6")
    flash_attn = ModuleType("flash_attn")
    flash_attn.__path__ = []
    flash_interface = ModuleType("flash_attn.flash_attn_interface")
    flash_triton = ModuleType("flash_attn.flash_attn_triton")
    flash_interface.flash_attn_func = lambda *args, **kwargs: None
    flash_interface.flash_attn_varlen_func = lambda *args, **kwargs: None
    flash_triton.flash_attn_func = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "flash_attn", flash_attn)
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.flash_attn_interface",
        flash_interface,
    )
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.flash_attn_triton",
        flash_triton,
    )

    neox_args = attention_args(
        hidden_size=8,
        num_attention_heads=2,
        num_kv_heads=1,
        head_dim=8,
        attention_config=[[["flash"], "all"]],
    )

    with model_parallel_context(1):
        attention = build_attention(neox_args)
        hidden_states = torch.randn(3, 2, 8)
        query, key, value = attention.gqa_project(hidden_states, attention_mask=None)

    assert query.shape == (3, 2, 2, 8)
    assert key.shape == (3, 2, 1, 8)
    assert value.shape == (3, 2, 1, 8)
