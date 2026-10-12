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
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


class FakeTEModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inp, *args, **kwargs):
        return inp


class FakeTEMultiheadAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.te_init_kwargs = kwargs
        self.hidden_size_per_attention_head = kwargs["kv_channels"]
        self.hidden_size_q = (
            kwargs["num_attention_heads"] * self.hidden_size_per_attention_head
        )
        num_gqa_groups = kwargs["num_gqa_groups"] or kwargs["num_attention_heads"]
        self.hidden_size_kv = (
            num_gqa_groups * self.hidden_size_per_attention_head
        )
        tp_size = kwargs["tp_size"]
        self.qkv = SimpleNamespace(
            weight=torch.empty(
                (self.hidden_size_q + 2 * self.hidden_size_kv) // tp_size,
                kwargs["hidden_size"],
            )
        )
        self.proj = SimpleNamespace(
            weight=torch.empty(
                kwargs["hidden_size"],
                self.hidden_size_q // tp_size,
            )
        )

    def forward(self, hidden_states, attention_mask, **kwargs):
        return hidden_states, None


class FakeRotaryEmbedding:
    def __init__(self, dim, **kwargs):
        self.dim = dim
        self.kwargs = kwargs

    def get_emb(self):
        return SimpleNamespace(dim=self.dim)


@pytest.fixture
def transformer_engine_wrapper(monkeypatch):
    fake_te = ModuleType("transformer_engine")
    fake_te.__version__ = "1.12.0"
    fake_te.pytorch = SimpleNamespace(
        RMSNorm=FakeTEModule,
        LayerNorm=FakeTEModule,
        Linear=FakeTEModule,
        LayerNormMLP=FakeTEModule,
        MultiheadAttention=FakeTEMultiheadAttention,
        fp8_autocast=lambda **kwargs: nullcontext(),
    )
    fake_te.common = SimpleNamespace(
        recipe=SimpleNamespace(
            DelayedScaling=FakeTEModule,
            Format=SimpleNamespace(E4M3="e4m3", HYBRID="hybrid"),
        )
    )

    module_name = "megatron.model.transformer_engine"
    previous_wrapper = sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "transformer_engine", fake_te)
    wrapper = importlib.import_module(module_name)
    try:
        yield wrapper
    finally:
        sys.modules.pop(module_name, None)
        if previous_wrapper is not None:
            sys.modules[module_name] = previous_wrapper


def te_attention_args(
    *,
    hidden_size,
    num_attention_heads,
    head_dim,
    num_kv_heads=None,
    pos_emb="none",
    rotary_pct=1.0,
):
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        pos_emb=pos_emb,
        rotary_pct=rotary_pct,
        rotary_emb_base=10000,
        rotary_save_freqs_buffer=False,
        attention_dropout=0.0,
        sliding_window_width=None,
        sequence_parallel=False,
        seq_length=8,
        train_micro_batch_size_per_gpu=1,
        params_dtype=torch.float32,
        norm="layernorm",
    )


def build_te_attention(monkeypatch, wrapper, neox_args, tp_size):
    monkeypatch.setattr(wrapper, "get_model_parallel_world_size", lambda: tp_size)
    monkeypatch.setattr(wrapper, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    return wrapper.TEMultiheadAttention(
        neox_args=neox_args,
        attention_mask_func=None,
        init_method=None,
        output_layer_init_method=None,
        layer_number=0,
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("hidden_size", "num_attention_heads", "num_kv_heads", "head_dim"),
    [
        (1024, 16, 8, 128),
        (8, 2, None, 6),
        (16, 4, 2, None),
    ],
)
@pytest.mark.parametrize("tp_size", [1, 2])
def test_te_attention_uses_effective_head_dim_for_projection_shapes(
    monkeypatch,
    transformer_engine_wrapper,
    hidden_size,
    num_attention_heads,
    num_kv_heads,
    head_dim,
    tp_size,
):
    neox_args = te_attention_args(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    attention = build_te_attention(
        monkeypatch,
        transformer_engine_wrapper,
        neox_args,
        tp_size,
    )

    effective_head_dim = head_dim or hidden_size // num_attention_heads
    query_hidden_size = num_attention_heads * effective_head_dim
    effective_num_kv_heads = num_kv_heads or num_attention_heads
    kv_hidden_size = effective_num_kv_heads * effective_head_dim

    assert attention.te_init_kwargs["kv_channels"] == effective_head_dim
    assert attention.te_init_kwargs["num_gqa_groups"] == num_kv_heads
    assert attention.hidden_size_per_attention_head == effective_head_dim
    assert attention.hidden_size_q == query_hidden_size
    assert attention.hidden_size_kv == kv_hidden_size
    assert attention.qkv.weight.shape == (
        (query_hidden_size + 2 * kv_hidden_size) // tp_size,
        hidden_size,
    )
    assert attention.proj.weight.shape == (
        hidden_size,
        query_hidden_size // tp_size,
    )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("head_dim", "rotary_pct", "expected_rotary_ndims", "expected_rope_dim"),
    [
        (None, 0.5, 2, 2),
        (8, 0.5, 4, 4),
        (8, 1.0, None, 8),
    ],
)
def test_te_rotary_uses_effective_head_dim(
    monkeypatch,
    transformer_engine_wrapper,
    head_dim,
    rotary_pct,
    expected_rotary_ndims,
    expected_rope_dim,
):
    monkeypatch.setattr(
        transformer_engine_wrapper,
        "RotaryEmbedding",
        FakeRotaryEmbedding,
    )
    neox_args = te_attention_args(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=head_dim,
        pos_emb="rotary",
        rotary_pct=rotary_pct,
    )

    attention = build_te_attention(
        monkeypatch,
        transformer_engine_wrapper,
        neox_args,
        tp_size=1,
    )

    assert attention.rotary_ndims == expected_rotary_ndims
    assert attention.rotary_embeddings.dim == expected_rope_dim


def test_real_te_attention_forward_backward_when_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Transformer Engine")
    pytest.importorskip("transformer_engine")
    wrapper = importlib.import_module("megatron.model.transformer_engine")
    monkeypatch.setattr(wrapper, "get_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(wrapper, "get_tensor_model_parallel_group", lambda: None)
    neox_args = te_attention_args(
        hidden_size=32,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        pos_emb="rotary",
    )
    neox_args.params_dtype = torch.float16
    init_method = lambda tensor: torch.nn.init.normal_(tensor, std=0.02)
    attention = wrapper.TEMultiheadAttention(
        neox_args=neox_args,
        attention_mask_func=None,
        init_method=init_method,
        output_layer_init_method=init_method,
        layer_number=0,
    )
    hidden_states = torch.randn(
        4,
        2,
        neox_args.hidden_size,
        device="cuda",
        dtype=neox_args.params_dtype,
        requires_grad=True,
    )

    output, _ = attention(hidden_states, None)
    output.float().square().mean().backward()

    assert output.shape == hidden_states.shape
    assert attention.qkv.weight.shape == (128, 32)
    assert attention.proj.weight.shape == (32, 64)
    assert attention.qkv.weight.grad is not None
    assert attention.proj.weight.grad is not None
    assert hidden_states.grad is not None
