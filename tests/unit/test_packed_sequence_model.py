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

from megatron.model.positional_embeddings import RotaryEmbedding
from megatron.model.transformer import ParallelSelfAttention

BOUNDARIES = (0, 3, 5)


def _packed_kwargs(boundaries):
    lengths = [end - start for start, end in zip(boundaries[:-1], boundaries[1:])]
    as_tensor = lambda value: torch.tensor(value, dtype=torch.int32, device="cuda")
    return dict(
        cu_seqlens=as_tensor(boundaries),
        num_documents=as_tensor(len(lengths)),
        max_seqlen=as_tensor(max(lengths)),
    )


def _assert_packed_matches_per_document(attention, hidden_states, atol):
    mask = torch.zeros(1, dtype=torch.bool, device="cuda")
    packed = attention(hidden_states, mask, **_packed_kwargs(BOUNDARIES))[0]
    reference = torch.cat(
        [
            attention(hidden_states[start:end], mask)[0]
            for start, end in zip(BOUNDARIES[:-1], BOUNDARIES[1:])
        ]
    )
    torch.testing.assert_close(packed, reference, atol=atol, rtol=atol)
    probe = torch.randn_like(packed)
    (packed_grad,) = torch.autograd.grad((packed * probe).sum(), hidden_states)
    (reference_grad,) = torch.autograd.grad((reference * probe).sum(), hidden_states)
    torch.testing.assert_close(
        packed_grad, reference_grad, atol=2 * atol, rtol=2 * atol
    )


def _make_flash_attention(mode, num_heads=2, head_dim=16):
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from flash_attn.layers.rotary import apply_rotary_emb

    class QKV(torch.nn.Module):
        def forward(self, hidden_states):
            shaped = hidden_states.view(*hidden_states.shape[:-1], num_heads, head_dim)
            mixed = torch.cat((shaped, shaped * 0.75 + 0.1, shaped * 1.25 - 0.2), -1)
            return mixed.flatten(start_dim=-2), None

    attention = ParallelSelfAttention.__new__(ParallelSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.__dict__.update(
        gqa=False,
        query_key_value=QKV(),
        dense=lambda hidden_states: (hidden_states, None),
        num_attention_heads_per_partition=num_heads,
        num_kv_heads_per_partition=num_heads,
        hidden_size_per_attention_head=head_dim,
        hidden_size_per_partition=num_heads * head_dim,
        use_qk_norm=False,
        use_cache=False,
        use_flash_attention=True,
        use_triton=False,
        sparse=False,
        dropout_p=0.0,
        sliding_window_width=1 if mode == "sliding-window" else None,
        pos_emb="alibi" if mode == "alibi" else "none",
        alibi_embed=SimpleNamespace(slopes=torch.tensor([0.5, 0.25], device="cuda")),
        flash_qkv_fn=flash_attn_func,
        flash_varlen_qkv_fn=flash_attn_varlen_func,
        flash_apply_rotary_fn=apply_rotary_emb,
        bf16=False,
        rope_fusion=False,
        rotary_ndims=head_dim // 2 if mode == "rotary-partial" else None,
        rotary_emb=None,
    )
    if mode.startswith("rotary"):
        attention.rotary_emb = RotaryEmbedding(
            attention.rotary_ndims or head_dim, max_seq_len=5, precision=torch.float16
        )
    return attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode", ["none", "rotary-full", "rotary-partial", "alibi", "sliding-window"]
)
def test_flash_packed_attention_matches_per_document(mode):
    pytest.importorskip("flash_attn")
    torch.manual_seed(1234)
    attention = _make_flash_attention(mode)
    hidden_states = torch.randn(5, 1, 32, device="cuda", dtype=torch.float16) * 0.2
    _assert_packed_matches_per_document(
        attention, hidden_states.requires_grad_(), atol=5e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "rotary_dim", [8, 4, None], ids=["rotary-full", "rotary-partial", "none"]
)
@pytest.mark.parametrize("num_gqa_groups", [None, 2], ids=["mha", "gqa"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no-bias"])
def test_te_packed_attention_matches_per_document(rotary_dim, num_gqa_groups, bias):
    pytest.importorskip("transformer_engine")
    from megatron.model.transformer_engine import TEMultiheadAttention, te

    torch.manual_seed(1234)
    attention = TEMultiheadAttention.__new__(TEMultiheadAttention)
    te.pytorch.MultiheadAttention.__init__(
        attention,
        hidden_size=32,
        num_attention_heads=4,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        device="cuda",
        qkv_format="sbhd",
        fuse_qkv_params=True,
        bias=bias,
        return_bias=bias,
    )
    attention.rope_emb = (
        rotary_dim
        and RotaryEmbedding(
            rotary_dim, max_seq_len=3, precision=torch.bfloat16
        ).get_emb()
    )
    hidden_states = torch.randn(5, 1, 32, device="cuda", dtype=torch.bfloat16) * 0.2
    _assert_packed_matches_per_document(
        attention, hidden_states.requires_grad_(), atol=1e-2
    )
