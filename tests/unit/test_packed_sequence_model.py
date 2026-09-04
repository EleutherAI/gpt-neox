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

import math
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.data.packed_sequence import PackedSequenceModelInputs
from megatron.model.gpt2_model import (
    _post_transformer_block,
    _pre_transformer_block,
)
from megatron.model.transformer import (
    ParallelSelfAttention,
    ParallelTransformerLayer,
    ParallelTransformerLayerPipe,
)
from megatron.model.positional_embeddings import RotaryEmbedding
from megatron.model.utils import SequentialWrapper
from megatron.model.word_embeddings import Embedding, EmbeddingPipe


@pytest.mark.cpu
@pytest.mark.parametrize("packed", [False, True], ids=["ordinary", "packed"])
def test_embedding_and_gpt_transforms_preserve_model_context(monkeypatch, packed):
    tokens = torch.tensor([[10, 11, 12, 13]])
    position_ids = torch.tensor([[0, 1, 0, 1]])
    hidden_states = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
    attention_mask = torch.zeros(1, dtype=torch.bool)

    if packed:
        model_inputs = PackedSequenceModelInputs(
            tokens=tokens,
            position_ids=position_ids,
            cu_seqlens=torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            num_documents=torch.tensor(2, dtype=torch.int32),
            max_seqlen=torch.tensor(2, dtype=torch.int32),
            attention_mask=attention_mask,
        )
    else:
        model_inputs = (tokens, position_ids, attention_mask)

    embedding = EmbeddingPipe.__new__(EmbeddingPipe)
    torch.nn.Module.__init__(embedding)

    def embed(_self, input_ids, positions, tokentype_ids=None):
        assert input_ids is tokens
        assert positions is position_ids
        return hidden_states

    monkeypatch.setattr(Embedding, "forward", embed)

    embedded = embedding(model_inputs)
    assert embedded[0] is hidden_states
    assert all(
        actual is expected
        for actual, expected in zip(embedded[1:], model_inputs[2:])
    )

    transformer_input = _pre_transformer_block(embedded)
    assert torch.equal(transformer_input[0], hidden_states.transpose(0, 1))
    assert all(
        actual is expected
        for actual, expected in zip(transformer_input[1:], embedded[1:])
    )

    output = _post_transformer_block(transformer_input)
    assert torch.equal(output, hidden_states)


@pytest.mark.cpu
@pytest.mark.parametrize("packed", [False, True], ids=["ordinary", "packed"])
def test_transformer_layer_pipe_threads_model_context(monkeypatch, packed):
    hidden_states = torch.zeros((4, 1, 8))
    attention_mask = torch.zeros(1, dtype=torch.bool)
    if packed:
        context = (
            torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            attention_mask,
        )
    else:
        context = (attention_mask,)

    received = []

    def forward(
        _self,
        hidden,
        mask,
        layer_past=None,
        *,
        cu_seqlens=None,
        num_documents=None,
        max_seqlen=None,
    ):
        received.append(
            (
                hidden,
                mask,
                cu_seqlens,
                num_documents,
                max_seqlen,
            )
        )
        return hidden + 1

    monkeypatch.setattr(ParallelTransformerLayer, "forward", forward)
    layers = []
    for _ in range(2):
        layer = ParallelTransformerLayerPipe.__new__(ParallelTransformerLayerPipe)
        torch.nn.Module.__init__(layer)
        layers.append(layer)

    output = (hidden_states, *context)
    for layer in layers:
        output = layer(output)

    expected_metadata = context[:-1] if packed else (None, None, None)
    assert len(received) == len(layers)
    for layer_index, values in enumerate(received):
        assert torch.equal(values[0], hidden_states + layer_index)
        assert all(
            actual is expected
            for actual, expected in zip(
                values[1:],
                (attention_mask, *expected_metadata),
            )
        )
    assert torch.equal(output[0], hidden_states + len(layers))
    assert all(
        actual is expected for actual, expected in zip(output[1:], context)
    )


@pytest.mark.cpu
def test_deepspeed_two_stage_transport_preserves_packed_context(monkeypatch):
    """Exercise DeepSpeed's stage send/receive logic without a network backend."""

    from deepspeed.runtime.pipe import engine as pipeline_engine

    packed_context = (
        torch.arange(8, dtype=torch.float32).view(4, 1, 2),
        torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
        torch.tensor(2, dtype=torch.int32),
        torch.ones(1, dtype=torch.bool),
    )
    mailbox = []

    def send(tensor, _stage):
        mailbox.append(tensor.detach().clone())

    monkeypatch.setattr(pipeline_engine.p2p, "send", send)
    sender = SimpleNamespace(
        wall_clock_breakdown=lambda: False,
        pipe_buffers={"outputs": [packed_context]},
        has_attention_mask=True,
        has_bool_tensors=False,
        dynamic_shape=False,
        first_output_send=False,
        next_stage=1,
    )

    pipeline_engine.PipelineEngine._exec_send_activations(sender, 0)

    assert [tensor.dtype for tensor in mailbox] == [
        torch.float32,
        torch.int32,
        torch.int32,
        torch.int32,
        torch.float16,
    ]

    def recv(tensor, _stage):
        tensor.copy_(mailbox.pop(0))

    monkeypatch.setattr(pipeline_engine.p2p, "recv", recv)
    receive_buffers = tuple(
        torch.empty_like(
            tensor,
            dtype=torch.float16 if tensor.dtype == torch.bool else tensor.dtype,
        )
        for tensor in packed_context
    )
    receiver = SimpleNamespace(
        wall_clock_breakdown=lambda: False,
        dynamic_shape=False,
        pipe_recv_buf=receive_buffers,
        prev_stage=0,
        is_pipe_partitioned=False,
        meta_buffer=None,
        device=torch.device("cpu"),
        has_attention_mask=True,
        has_bool_tensors=False,
        pipe_buffers={"inputs": [None]},
    )

    pipeline_engine.PipelineEngine._exec_recv_activations(receiver, 0)

    received = receiver.pipe_buffers["inputs"][0]
    assert not mailbox
    assert len(received) == len(packed_context)
    for actual, expected in zip(received, packed_context):
        assert torch.equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape


@pytest.mark.cpu
@pytest.mark.parametrize("gpt_j_residual", [False, True])
def test_transformer_layer_passes_metadata_to_attention(gpt_j_residual):
    class Attention(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            attention_mask,
            layer_past=None,
            *,
            cu_seqlens=None,
            num_documents=None,
            max_seqlen=None,
        ):
            self.received = (
                attention_mask,
                cu_seqlens,
                num_documents,
                max_seqlen,
            )
            return torch.zeros_like(hidden_states), None

    class MLP(torch.nn.Module):
        def forward(self, hidden_states):
            return torch.zeros_like(hidden_states), None

    layer = ParallelTransformerLayer.__new__(ParallelTransformerLayer)
    torch.nn.Module.__init__(layer)
    layer.neox_args = SimpleNamespace(
        te_fp8_mha=False,
        te_layernorm_mlp=False,
    )
    layer.layer_past = None
    layer.hidden_dropout = 0.0
    layer.bias_dropout_fusion = False
    layer.gpt_j_residual = gpt_j_residual
    layer.gpt_j_tied = True
    layer.use_cache = False
    layer.num_experts = 1
    layer.input_layernorm = torch.nn.Identity()
    layer.post_attention_layernorm = torch.nn.Identity()
    layer.attention = Attention()
    layer.mlp = MLP()
    layer.reduce = lambda value: value
    layer.eval()

    hidden_states = torch.zeros((4, 1, 8))
    attention_mask = torch.zeros(1, dtype=torch.bool)
    cu_seqlens = torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32)
    num_documents = torch.tensor(2, dtype=torch.int32)
    max_seqlen = torch.tensor(2, dtype=torch.int32)

    output = layer(
        hidden_states,
        attention_mask,
        cu_seqlens=cu_seqlens,
        num_documents=num_documents,
        max_seqlen=max_seqlen,
    )

    assert torch.equal(output, hidden_states)
    assert all(
        actual is expected
        for actual, expected in zip(
            layer.attention.received,
            (attention_mask, cu_seqlens, num_documents, max_seqlen),
        )
    )


@pytest.mark.cpu
def test_te_attention_dispatches_packed_thd_without_mutating_fixed_format(
    monkeypatch,
):
    pytest.importorskip("transformer_engine")
    from megatron.model.transformer_engine import TEMultiheadAttention, te

    attention = TEMultiheadAttention.__new__(TEMultiheadAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_format = "sbhd"
    attention.use_cache = False
    attention.pos_emb = "alibi"
    attention.packed_window_size = (3, 0)
    attention.alibi_embed = SimpleNamespace(slopes=torch.tensor([0.5]))
    attention.probe = torch.nn.Parameter(torch.ones(4))
    calls = []

    def te_forward(module, hidden_states, attention_mask=None, **kwargs):
        calls.append((module, hidden_states, attention_mask, kwargs))
        output = (
            hidden_states.squeeze(1)
            if module.qkv_format == "thd"
            else hidden_states
        )
        return output, module.probe

    monkeypatch.setattr(te.pytorch.MultiheadAttention, "forward", te_forward)

    hidden_states = torch.zeros((5, 1, 4), dtype=torch.bfloat16)
    attention_mask = torch.zeros(1, dtype=torch.bool)
    padded_cu_seqlens = torch.tensor(
        [0, 2, 5, 5, 5, 5], dtype=torch.int32
    )
    packed_output, packed_bias = attention(
        hidden_states,
        attention_mask,
        cu_seqlens=padded_cu_seqlens,
        num_documents=torch.tensor(2, dtype=torch.int32),
        max_seqlen=torch.tensor(3, dtype=torch.int32),
        is_first_microbatch=True,
    )

    packed_module, packed_hidden_states, packed_mask, packed_kwargs = calls.pop(0)
    assert packed_module is not attention
    assert packed_module.probe is attention.probe
    assert packed_module.qkv_format == "thd"
    assert attention.qkv_format == "sbhd"
    assert packed_mask is None
    assert packed_hidden_states.dtype == torch.bfloat16
    assert packed_output.shape == hidden_states.shape
    assert packed_bias is attention.probe
    assert packed_kwargs["attn_mask_type"] == "padding_causal"
    assert packed_kwargs["window_size"] == (3, 0)
    assert torch.equal(
        packed_kwargs["cu_seqlens_q"],
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    assert packed_kwargs["cu_seqlens_kv"] is packed_kwargs["cu_seqlens_q"]
    assert packed_kwargs["cu_seqlens_q"].dtype == torch.int32
    assert packed_kwargs["max_seqlen_q"] == 3
    assert packed_kwargs["max_seqlen_kv"] == 3
    assert isinstance(packed_kwargs["max_seqlen_q"], int)
    assert packed_kwargs["core_attention_bias_type"] == "alibi"
    assert torch.equal(packed_kwargs["alibi_slopes"], torch.tensor([0.5]))
    assert packed_kwargs["alibi_slopes"].dtype == torch.float32
    assert packed_kwargs["is_first_microbatch"] is True

    fixed_output, fixed_bias = attention(hidden_states, attention_mask)

    fixed_module, _, fixed_mask, fixed_kwargs = calls.pop(0)
    assert fixed_module is attention
    assert fixed_module.qkv_format == "sbhd"
    assert fixed_mask is attention_mask
    assert fixed_output is hidden_states
    assert fixed_bias is attention.probe
    assert "cu_seqlens_q" not in fixed_kwargs
    assert "attn_mask_type" not in fixed_kwargs
    assert not calls


@pytest.mark.cpu
def test_te_attention_rejects_kv_cache_for_packed_execution():
    pytest.importorskip("transformer_engine")
    from megatron.model.transformer_engine import TEMultiheadAttention

    attention = TEMultiheadAttention.__new__(TEMultiheadAttention)
    torch.nn.Module.__init__(attention)
    attention.qkv_format = "sbhd"
    attention.use_cache = False
    attention.pos_emb = "none"
    attention.packed_window_size = None
    packed_args = {
        "cu_seqlens": torch.tensor([0, 2], dtype=torch.int32),
        "num_documents": torch.tensor(1, dtype=torch.int32),
        "max_seqlen": torch.tensor(2, dtype=torch.int32),
    }

    incompatible_cache_args = (
        {"layer_past": torch.ones(1)},
        {"inference_params": object()},
    )
    for cache_args in incompatible_cache_args:
        with pytest.raises(RuntimeError, match="does not support KV-cache"):
            attention(
                torch.zeros((2, 1, 4)),
                torch.zeros(1, dtype=torch.bool),
                **packed_args,
                **cache_args,
            )

    attention.use_cache = True
    with pytest.raises(RuntimeError, match="does not support KV-cache"):
        attention(
            torch.zeros((2, 1, 4)),
            torch.zeros(1, dtype=torch.bool),
            **packed_args,
        )


def _make_cuda_te_attention(mode, max_seq_len=3, num_gqa_groups=None):
    from megatron.model.transformer_engine import TEMultiheadAttention, te

    window_size = (1, 0) if mode == "sliding-window" else None
    num_attention_heads = 4 if num_gqa_groups is not None else 2
    head_dim = 32 // num_attention_heads
    attention = TEMultiheadAttention.__new__(TEMultiheadAttention)
    te.pytorch.MultiheadAttention.__init__(
        attention,
        hidden_size=32,
        num_attention_heads=num_attention_heads,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        device="cuda",
        qkv_format="sbhd",
        window_size=window_size,
        fuse_qkv_params=True,
        return_bias=True,
    )
    attention.use_cache = False
    attention.pos_emb = (
        "rotary" if mode.startswith("rotary") else "alibi" if mode == "alibi" else mode
    )
    attention.packed_window_size = window_size

    if mode.startswith("rotary"):
        rotary_dim = head_dim // 2 if mode == "rotary-partial" else head_dim
        attention.rope_emb = RotaryEmbedding(
            rotary_dim,
            max_seq_len=max_seq_len,
            precision=torch.bfloat16,
        ).get_emb()
    elif mode == "alibi":
        attention.alibi_embed = SimpleNamespace(
            slopes=torch.tensor(
                [0.5 ** (index + 1) for index in range(num_attention_heads)],
                device="cuda",
            )
        )

    attention.train()
    return attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode",
    ["learned", "rotary-full", "rotary-partial", "alibi", "sliding-window"],
)
@pytest.mark.parametrize(
    "num_gqa_groups",
    [None, 2],
    ids=["mha", "gqa"],
)
def test_te_packed_thd_matches_per_document_reference(mode, num_gqa_groups):
    pytest.importorskip("transformer_engine")

    torch.manual_seed(1234)
    attention = _make_cuda_te_attention(mode, num_gqa_groups=num_gqa_groups)
    boundaries = (0, 64, 128) if mode == "alibi" else (0, 2, 5)
    total_tokens = boundaries[-1]
    max_seqlen = max(end - start for start, end in zip(boundaries[:-1], boundaries[1:]))

    if mode == "learned":
        embedding = _make_cuda_learned_embedding(dtype=torch.bfloat16)
        token_ids = torch.zeros((1, total_tokens), dtype=torch.long, device="cuda")
        position_ids = torch.tensor([[0, 1, 0, 1, 2]], dtype=torch.long, device="cuda")
        hidden_states = embedding(token_ids, position_ids).transpose(0, 1)
        gradient_targets = tuple(embedding.parameters())
    else:
        hidden_states = (
            torch.randn(
                (total_tokens, 1, 32),
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.2
        ).requires_grad_()
        gradient_targets = (hidden_states,)
    attention_parameters = tuple(attention.parameters())
    gradient_targets += attention_parameters

    attention_mask = torch.zeros(1, dtype=torch.bool, device="cuda")
    packed_output, packed_bias = attention(
        hidden_states,
        attention_mask,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device="cuda"),
        num_documents=torch.tensor(2, dtype=torch.int32, device="cuda"),
        max_seqlen=torch.tensor(max_seqlen, dtype=torch.int32, device="cuda"),
    )

    reference_outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        reference_output, reference_bias = attention(
            hidden_states[start:end], attention_mask
        )
        reference_outputs.append(reference_output)
        assert torch.equal(reference_bias, packed_bias)
    reference_output = torch.cat(reference_outputs)

    assert attention.qkv_format == "sbhd"
    assert packed_output.shape == hidden_states.shape
    torch.testing.assert_close(packed_output, reference_output, atol=1e-2, rtol=1e-2)

    probe = torch.randn_like(packed_output)
    packed_grads = torch.autograd.grad(
        (packed_output.float() * probe.float()).sum(),
        gradient_targets,
        retain_graph=True,
        allow_unused=True,
    )
    reference_grads = torch.autograd.grad(
        (reference_output.float() * probe.float()).sum(),
        gradient_targets,
        allow_unused=True,
    )
    assert any(
        gradient is not None
        for gradient in packed_grads[-len(attention_parameters) :]
    )
    for packed_grad, reference_grad in zip(packed_grads, reference_grads):
        if packed_grad is None or reference_grad is None:
            assert packed_grad is reference_grad
            continue
        torch.testing.assert_close(packed_grad, reference_grad, atol=2e-2, rtol=2e-2)

    changed_first_document = hidden_states.detach().clone()
    changed_first_document[: boundaries[1]].add_(1.0)
    changed_output, _ = attention(
        changed_first_document,
        attention_mask,
        cu_seqlens=torch.tensor(boundaries, dtype=torch.int32, device="cuda"),
        num_documents=torch.tensor(2, dtype=torch.int32, device="cuda"),
        max_seqlen=torch.tensor(max_seqlen, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        changed_output[boundaries[1] :],
        packed_output.detach()[boundaries[1] :],
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_partial_rotary_thd_preserves_nonrotary_suffix():
    pytest.importorskip("transformer_engine")
    from transformer_engine.pytorch.attention import apply_rotary_pos_emb

    torch.manual_seed(1234)
    rotary_dim = 8
    hidden_states = torch.randn((5, 2, 16), device="cuda", dtype=torch.bfloat16)
    rope_emb = RotaryEmbedding(
        rotary_dim,
        max_seq_len=3,
        precision=torch.bfloat16,
    ).get_emb()
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")

    packed_output = apply_rotary_pos_emb(
        hidden_states,
        rope_emb,
        tensor_format="thd",
        fused=True,
        cu_seqlens=cu_seqlens,
    )
    reference_output = torch.cat(
        [
            apply_rotary_pos_emb(
                hidden_states[start:end].unsqueeze(1),
                rope_emb,
                tensor_format="sbhd",
            ).squeeze(1)
            for start, end in ((0, 2), (2, 5))
        ]
    )

    torch.testing.assert_close(
        packed_output[..., :rotary_dim],
        reference_output[..., :rotary_dim],
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.equal(packed_output[..., rotary_dim:], hidden_states[..., rotary_dim:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_fixed_evaluation_is_unchanged_after_packed_training_forward():
    pytest.importorskip("transformer_engine")

    torch.manual_seed(1234)
    attention = _make_cuda_te_attention("rotary-partial", max_seq_len=8)
    attention_mask = torch.zeros(1, dtype=torch.bool, device="cuda")
    fixed_hidden_states = torch.randn(
        (4, 3, 32),
        device="cuda",
        dtype=torch.bfloat16,
    )

    attention.eval()
    reference_output, reference_bias = attention(
        fixed_hidden_states,
        attention_mask,
    )
    parameter_names = tuple(attention.state_dict())

    attention.train()
    attention(
        torch.randn((5, 1, 32), device="cuda", dtype=torch.bfloat16),
        attention_mask,
        cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda"),
        num_documents=torch.tensor(2, dtype=torch.int32, device="cuda"),
        max_seqlen=torch.tensor(3, dtype=torch.int32, device="cuda"),
    )

    attention.eval()
    output, bias = attention(fixed_hidden_states, attention_mask)

    assert attention.qkv_format == "sbhd"
    assert output.shape == fixed_hidden_states.shape
    assert tuple(attention.state_dict()) == parameter_names
    torch.testing.assert_close(output, reference_output, atol=0, rtol=0)
    torch.testing.assert_close(bias, reference_bias, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_te_packed_checkpointing_handles_consecutive_microbatches():
    pytest.importorskip("transformer_engine")

    torch.manual_seed(1234)
    attention = _make_cuda_te_attention("rotary-partial", max_seq_len=4)
    attention_mask = torch.zeros(1, dtype=torch.bool, device="cuda")
    microbatch_boundaries = ((0, 2, 5), (0, 1, 2, 6))

    for boundaries in microbatch_boundaries:
        total_tokens = boundaries[-1]
        max_seqlen = max(
            end - start for start, end in zip(boundaries[:-1], boundaries[1:])
        )
        hidden_states = torch.randn(
            (total_tokens, 1, 32),
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        cu_seqlens = torch.tensor(
            boundaries,
            dtype=torch.int32,
            device="cuda",
        )
        num_documents = torch.tensor(
            len(boundaries) - 1,
            dtype=torch.int32,
            device="cuda",
        )
        max_seqlen_tensor = torch.tensor(
            max_seqlen,
            dtype=torch.int32,
            device="cuda",
        )

        with torch.no_grad():
            reference_output = torch.cat(
                [
                    attention(hidden_states[start:end], attention_mask)[0]
                    for start, end in zip(boundaries[:-1], boundaries[1:])
                ]
            )

        def packed_forward(states, packed_boundaries, document_count, maximum):
            return attention(
                states,
                attention_mask,
                cu_seqlens=packed_boundaries,
                num_documents=document_count,
                max_seqlen=maximum,
            )[0]

        output = checkpoint(
            packed_forward,
            hidden_states,
            cu_seqlens,
            num_documents,
            max_seqlen_tensor,
            use_reentrant=True,
        )
        torch.testing.assert_close(output, reference_output, atol=1e-2, rtol=1e-2)

        output.float().square().mean().backward()
        assert hidden_states.grad is not None
        assert torch.isfinite(hidden_states.grad).all()
        attention.zero_grad(set_to_none=True)


def _make_native_flash_attention(*, rotary_ndims=None):
    """Construct a CPU-only attention shell around mocked FlashAttention ops."""

    class QKV(torch.nn.Module):
        def forward(self, hidden_states):
            return (
                torch.cat(
                    (hidden_states, hidden_states + 10, hidden_states + 20), dim=-1
                ),
                None,
            )

    class Dense(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    attention = ParallelSelfAttention.__new__(ParallelSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.gqa = False
    attention.query_key_value = QKV()
    attention.dense = Dense()
    attention.num_attention_heads_per_partition = 1
    attention.num_kv_heads_per_partition = 1
    attention.hidden_size_per_attention_head = 4
    attention.hidden_size_per_partition = 4
    attention.use_qk_layernorm = False
    attention.rotary_ndims = rotary_ndims
    attention.use_cache = False
    attention.use_flash_attention = True
    attention.use_triton = False
    attention.sparse = False
    attention.dropout_p = 0.25
    attention.sliding_window_width = None
    attention.pos_emb = "none"
    return attention


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("training", "expected_dropout"),
    [(True, 0.25), (False, 0.0)],
    ids=["training", "evaluation"],
)
def test_native_attention_dispatches_packed_metadata_to_flash_varlen(
    training, expected_dropout
):
    attention = _make_native_flash_attention()
    attention.rotary_emb = None
    attention.sliding_window_width = 3
    attention.pos_emb = "alibi"
    attention.alibi_embed = SimpleNamespace(slopes=torch.tensor([0.5]))
    calls = []

    def flash_varlen(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    ):
        calls.append(
            (
                query,
                key,
                value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                kwargs,
            )
        )
        return query + key + value

    attention.flash_varlen_qkv_fn = flash_varlen
    attention.flash_qkv_fn = lambda *args, **kwargs: pytest.fail(
        "fixed-shape FlashAttention was called"
    )
    attention.train(training)

    hidden_states = torch.ones((5, 1, 4))
    padded_cu_seqlens = torch.tensor([0, 2, 5, 5, 5, 5], dtype=torch.int32)
    output, bias = attention(
        hidden_states,
        torch.zeros(1, dtype=torch.bool),
        cu_seqlens=padded_cu_seqlens,
        num_documents=torch.tensor(2, dtype=torch.int32),
        max_seqlen=torch.tensor(3, dtype=torch.int32),
    )

    assert bias is None
    assert output.shape == hidden_states.shape
    assert len(calls) == 1
    query, key, value, cu_q, cu_k, max_q, max_k, kwargs = calls[0]
    assert query.shape == key.shape == value.shape == (5, 1, 4)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert torch.equal(cu_q, torch.tensor([0, 2, 5], dtype=torch.int32))
    assert torch.equal(cu_k, cu_q)
    assert max_q == max_k == 3
    assert kwargs["dropout_p"] == expected_dropout
    assert kwargs["softmax_scale"] is None
    assert kwargs["causal"] is True
    assert kwargs["window_size"] == (3, -1)
    assert torch.equal(kwargs["alibi_slopes"], torch.tensor([0.5]))
    assert kwargs["alibi_slopes"].dtype == torch.float32


@pytest.mark.cpu
@pytest.mark.parametrize("rotary_ndims", [None, 2], ids=["full", "partial"])
def test_native_packed_rotary_resets_at_document_boundaries(rotary_ndims):
    attention = _make_native_flash_attention(rotary_ndims=rotary_ndims)
    attention.pos_emb = "rotary"
    attention.rope_fusion = True
    rotary_dim = rotary_ndims or attention.hidden_size_per_attention_head
    rotary_calls = []
    flash_calls = []

    class RotaryCache(torch.nn.Module):
        def forward(self, value_layer, seq_dim=0, seq_len=None):
            assert value_layer.shape == (5, 1, 1, 4)
            assert seq_len == 3
            values = torch.arange(seq_len * rotary_dim, dtype=torch.float32)
            values = values.view(seq_len, 1, 1, rotary_dim)
            return values.cos(), values.sin()

    def apply_packed_rotary(tensor, cos, sin, *, cu_seqlens, max_seqlen):
        rotary_calls.append((tensor, cos, sin, cu_seqlens, max_seqlen))
        local_positions = torch.empty(tensor.size(0), dtype=tensor.dtype)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            start, end = int(start), int(end)
            local_positions[start:end] = torch.arange(end - start, dtype=tensor.dtype)
        return tensor + local_positions[:, None, None]

    def flash_varlen(query, key, value, *args, **kwargs):
        flash_calls.append((query, key, value, args, kwargs))
        return query

    attention.rotary_emb = RotaryCache()
    attention.flash_apply_rotary_fn = apply_packed_rotary
    attention.flash_varlen_qkv_fn = flash_varlen
    attention.train()

    attention(
        torch.ones((5, 1, 4)),
        torch.zeros(1, dtype=torch.bool),
        cu_seqlens=torch.tensor([0, 2, 5, 5, 5, 5], dtype=torch.int32),
        num_documents=torch.tensor(2, dtype=torch.int32),
        max_seqlen=torch.tensor(3, dtype=torch.int32),
    )

    assert len(rotary_calls) == 2
    for tensor, cos, sin, cu_seqlens, max_seqlen in rotary_calls:
        assert tensor.shape == (5, 1, rotary_dim)
        assert cos.shape == sin.shape == (3, rotary_dim // 2)
        assert torch.equal(cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
        assert max_seqlen == 3

    assert len(flash_calls) == 1
    query, key, _, _, _ = flash_calls[0]
    expected_positions = torch.tensor([0, 1, 0, 1, 2], dtype=query.dtype)
    assert torch.equal(query[:, 0, 0], 1 + expected_positions)
    assert torch.equal(key[:, 0, 0], 11 + expected_positions)
    if rotary_ndims is not None:
        assert torch.equal(query[:, 0, -1], torch.ones(5))
        assert torch.equal(key[:, 0, -1], torch.full((5,), 11.0))


@pytest.mark.cpu
def test_native_flash_attention_preserves_fixed_shape_training_path():
    attention = _make_native_flash_attention()
    attention.pos_emb = "none"
    fixed_calls = []

    def flash_fixed(query, key, value, *args, **kwargs):
        fixed_calls.append((query, key, value, args, kwargs))
        return query

    attention.flash_qkv_fn = flash_fixed
    attention.flash_varlen_qkv_fn = lambda *args, **kwargs: pytest.fail(
        "varlen FlashAttention was called"
    )
    attention.train()

    query = torch.zeros((3, 2, 1, 4))
    output = attention.flash_attention(query, query, query)

    assert output.shape == (2, 1, 3, 4)
    assert len(fixed_calls) == 1
    assert fixed_calls[0][0].shape == (2, 3, 1, 4)
    assert fixed_calls[0][4]["causal"] is True


@pytest.mark.cpu
def test_native_flash_attention_preserves_ordinary_evaluation_path():
    attention = _make_native_flash_attention()
    attention.pos_emb = "none"
    varlen_calls = []

    def flash_varlen(query, key, value, *args, **kwargs):
        varlen_calls.append((query, key, value, args, kwargs))
        return query

    attention.flash_qkv_fn = lambda *args, **kwargs: pytest.fail(
        "fixed-shape training FlashAttention was called"
    )
    attention.flash_varlen_qkv_fn = flash_varlen
    attention.eval()

    query = torch.zeros((3, 2, 1, 4))
    output = attention.flash_attention(query, query, query)

    assert output.shape == (2, 1, 3, 4)
    assert len(varlen_calls) == 1
    _, _, _, args, kwargs = varlen_calls[0]
    cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = args
    expected_cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    assert torch.equal(cu_seqlens_q, expected_cu_seqlens)
    assert torch.equal(cu_seqlens_k, expected_cu_seqlens)
    assert max_seqlen_q == max_seqlen_k == 3
    assert kwargs["causal"] is True


@pytest.mark.cpu
def test_learned_embeddings_consume_document_local_position_ids():
    embedding = Embedding.__new__(Embedding)
    torch.nn.Module.__init__(embedding)
    embedding.word_embeddings = torch.nn.Embedding(2, 4)
    embedding.position_embeddings = torch.nn.Embedding(3, 4)
    embedding.use_pos_emb = True
    embedding.embedding_type = "learned"
    embedding.opt_pos_emb_offset = 0
    embedding.mup_rp_embedding_mult = 1.0
    embedding.tokentype_embeddings = None
    embedding.embedding_dropout = torch.nn.Identity()
    embedding.use_mup = False
    embedding.sequence_parallel = False

    with torch.no_grad():
        embedding.word_embeddings.weight.zero_()
        embedding.position_embeddings.weight.copy_(
            torch.arange(12, dtype=torch.float32).view(3, 4)
        )

    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    output = embedding(torch.zeros_like(position_ids), position_ids)

    assert torch.equal(output[0, 0], output[0, 3])
    assert torch.equal(output[0, 1], output[0, 4])
    assert torch.equal(output, embedding.position_embeddings(position_ids))

    output.sum().backward()
    expected_position_grad = torch.tensor([[2.0] * 4, [2.0] * 4, [1.0] * 4])
    assert torch.equal(
        embedding.position_embeddings.weight.grad, expected_position_grad
    )


def _make_cuda_native_flash_attention(mode):
    from flash_attn.flash_attn_interface import (
        flash_attn_func,
        flash_attn_varlen_func,
    )
    from flash_attn.layers.rotary import apply_rotary_emb

    num_heads = 2
    head_dim = 16

    class QKV(torch.nn.Module):
        def forward(self, hidden_states):
            shaped = hidden_states.view(
                *hidden_states.shape[:-1], num_heads, head_dim
            )
            mixed = torch.cat(
                (shaped, shaped * 0.75 + 0.1, shaped * 1.25 - 0.2), dim=-1
            )
            return mixed.flatten(start_dim=-2), None

    class Dense(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    attention = ParallelSelfAttention.__new__(ParallelSelfAttention)
    torch.nn.Module.__init__(attention)
    attention.gqa = False
    attention.query_key_value = QKV()
    attention.dense = Dense()
    attention.num_attention_heads_per_partition = num_heads
    attention.num_kv_heads_per_partition = num_heads
    attention.hidden_size_per_attention_head = head_dim
    attention.hidden_size_per_partition = num_heads * head_dim
    attention.use_qk_layernorm = False
    attention.use_cache = False
    attention.use_flash_attention = True
    attention.use_triton = False
    attention.sparse = False
    attention.dropout_p = 0.0
    attention.sliding_window_width = 1 if mode == "sliding-window" else None
    attention.pos_emb = "alibi" if mode == "alibi" else "none"
    attention.flash_qkv_fn = flash_attn_func
    attention.flash_varlen_qkv_fn = flash_attn_varlen_func
    attention.bf16 = False
    attention.rope_fusion = False

    if mode == "alibi":
        attention.alibi_embed = SimpleNamespace(
            slopes=torch.tensor([0.5, 0.25], device="cuda")
        )

    if mode.startswith("rotary"):
        attention.rotary_ndims = 8 if mode == "rotary-partial" else None
        rotary_dim = attention.rotary_ndims or head_dim
        attention.rotary_emb = RotaryEmbedding(
            rotary_dim,
            max_seq_len=5,
            precision=torch.float16,
        )
        attention.flash_apply_rotary_fn = apply_rotary_emb
    else:
        attention.rotary_ndims = None
        attention.rotary_emb = None

    return attention


def _make_cuda_learned_embedding(dtype=torch.float16):
    embedding = Embedding.__new__(Embedding)
    torch.nn.Module.__init__(embedding)
    embedding.word_embeddings = torch.nn.Embedding(
        2, 32, device="cuda", dtype=dtype
    )
    embedding.position_embeddings = torch.nn.Embedding(
        3, 32, device="cuda", dtype=dtype
    )
    embedding.use_pos_emb = True
    embedding.embedding_type = "learned"
    embedding.opt_pos_emb_offset = 0
    embedding.mup_rp_embedding_mult = 1.0
    embedding.tokentype_embeddings = None
    embedding.embedding_dropout = torch.nn.Identity()
    embedding.use_mup = False
    embedding.sequence_parallel = False
    return embedding


def _run_per_document_reference(attention, hidden_states, boundaries, mask):
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        output, bias = attention(hidden_states[start:end], mask)
        assert bias is None
        outputs.append(output)
    return torch.cat(outputs, dim=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode",
    ["learned", "rotary-full", "rotary-partial", "alibi", "sliding-window"],
)
def test_packed_positional_modes_match_real_flash_attention_reference(mode):
    pytest.importorskip("flash_attn")
    torch.manual_seed(1234)
    attention = _make_cuda_native_flash_attention(mode)
    attention.train()

    boundaries = (0, 3, 5)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
    num_documents = torch.tensor(2, dtype=torch.int32, device="cuda")
    max_seqlen = torch.tensor(3, dtype=torch.int32, device="cuda")
    attention_mask = torch.zeros(1, dtype=torch.bool, device="cuda")

    if mode == "learned":
        embedding = _make_cuda_learned_embedding()
        token_ids = torch.zeros((1, 5), dtype=torch.long, device="cuda")
        position_ids = torch.tensor(
            [[0, 1, 2, 0, 1]], dtype=torch.long, device="cuda"
        )
        hidden_states = embedding(token_ids, position_ids).transpose(0, 1)
        assert torch.equal(hidden_states[0], hidden_states[3])
        assert torch.equal(hidden_states[1], hidden_states[4])
        gradient_targets = tuple(embedding.parameters())
    else:
        hidden_states = (
            torch.randn((5, 1, 32), device="cuda", dtype=torch.float16) * 0.2
        ).requires_grad_()
        gradient_targets = (hidden_states,)

    packed_output, packed_bias = attention(
        hidden_states,
        attention_mask,
        cu_seqlens=cu_seqlens,
        num_documents=num_documents,
        max_seqlen=max_seqlen,
    )
    reference_output = _run_per_document_reference(
        attention, hidden_states, boundaries, attention_mask
    )

    assert packed_bias is None
    torch.testing.assert_close(
        packed_output, reference_output, atol=5e-3, rtol=5e-3
    )

    probe = torch.randn_like(packed_output)
    packed_grads = torch.autograd.grad(
        (packed_output.float() * probe.float()).sum(),
        gradient_targets,
        retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        (reference_output.float() * probe.float()).sum(), gradient_targets
    )
    for packed_grad, reference_grad in zip(packed_grads, reference_grads):
        torch.testing.assert_close(
            packed_grad, reference_grad, atol=1e-2, rtol=1e-2
        )

    changed_first_document = hidden_states.detach().clone()
    changed_first_document[: boundaries[1]].add_(1.0)
    changed_output, _ = attention(
        changed_first_document,
        attention_mask,
        cu_seqlens=cu_seqlens,
        num_documents=num_documents,
        max_seqlen=max_seqlen,
    )
    torch.testing.assert_close(
        changed_output[boundaries[1] :],
        packed_output.detach()[boundaries[1] :],
        atol=0,
        rtol=0,
    )

    attention.eval()
    evaluation_output, _ = attention(hidden_states.detach(), attention_mask)
    attention.train()
    training_output, _ = attention(hidden_states.detach(), attention_mask)
    torch.testing.assert_close(
        evaluation_output, training_output, atol=5e-3, rtol=5e-3
    )


def _dense_block_diagonal_attention(query, key, value, document_lengths):
    query = query.float()
    key = key.float()
    value = value.float()
    total_tokens = query.size(0)
    document_ids = torch.repeat_interleave(
        torch.arange(len(document_lengths), device=query.device),
        torch.tensor(document_lengths, device=query.device),
    )
    document_starts = torch.repeat_interleave(
        torch.tensor(
            [0, *torch.tensor(document_lengths).cumsum(0)[:-1].tolist()],
            device=query.device,
        ),
        torch.tensor(document_lengths, device=query.device),
    )
    local_positions = torch.arange(total_tokens, device=query.device) - document_starts
    block_diagonal_causal_mask = (document_ids[:, None] == document_ids[None, :]) & (
        local_positions[:, None] >= local_positions[None, :]
    )

    attention_scores = torch.einsum("thd,shd->hts", query, key)
    attention_scores /= math.sqrt(query.size(-1))
    attention_scores.masked_fill_(~block_diagonal_causal_mask.unsqueeze(0), -torch.inf)
    attention_probs = torch.softmax(attention_scores, dim=-1)
    return torch.einsum("hts,shd->thd", attention_probs, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tensor_parallel_size", [1, 2, 4])
def test_packed_flash_attention_head_partitions_match_dense_reference(
    tensor_parallel_size,
):
    pytest.importorskip("flash_attn")
    torch.manual_seed(5678)
    attention = _make_cuda_native_flash_attention("learned")
    document_lengths = (3, 5)
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device="cuda")

    query = (
        torch.randn((8, 1, 4, 16), dtype=torch.float16, device="cuda") * 0.2
    ).requires_grad_()
    key = (
        torch.randn((8, 1, 4, 16), dtype=torch.float16, device="cuda") * 0.2
    ).requires_grad_()
    value = (
        torch.randn((8, 1, 4, 16), dtype=torch.float16, device="cuda") * 0.2
    ).requires_grad_()

    partition_outputs = [
        attention._flash_attention_packed(
            query_partition,
            key_partition,
            value_partition,
            cu_seqlens=cu_seqlens,
            max_seqlen=5,
        )
        .squeeze(0)
        .transpose(0, 1)
        for query_partition, key_partition, value_partition in zip(
            query.chunk(tensor_parallel_size, dim=2),
            key.chunk(tensor_parallel_size, dim=2),
            value.chunk(tensor_parallel_size, dim=2),
        )
    ]
    packed_output = torch.cat(partition_outputs, dim=1)
    dense_output = _dense_block_diagonal_attention(
        query.squeeze(1),
        key.squeeze(1),
        value.squeeze(1),
        document_lengths,
    )

    torch.testing.assert_close(
        packed_output.float(), dense_output, atol=2e-3, rtol=2e-3
    )

    probe = torch.randn_like(dense_output)
    packed_grads = torch.autograd.grad(
        (packed_output.float() * probe).sum(),
        (query, key, value),
        retain_graph=True,
    )
    dense_grads = torch.autograd.grad(
        (dense_output * probe).sum(), (query, key, value)
    )
    for packed_grad, dense_grad in zip(packed_grads, dense_grads):
        torch.testing.assert_close(
            packed_grad.float(), dense_grad.float(), atol=3e-3, rtol=3e-3
        )


@pytest.mark.cpu
def test_sequential_wrapper_preserves_packed_context_during_checkpointing():
    """Exercise the wrapper constructed by GPT2ModelPipe.to_sequential()."""

    class CheckpointedParallelTransformerLayerPipe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.received_contexts = []

        def forward(self, args):
            hidden_states, *context = args
            self.received_contexts.append(
                tuple(tensor.detach().clone() for tensor in context)
            )
            return (hidden_states * self.weight, *context)

    class PostTransformerBlock(torch.nn.Module):
        def forward(self, args):
            self.received_context = tuple(
                tensor.detach().clone() for tensor in args[1:]
            )
            return _post_transformer_block(args)

    def run(checkpoint_interval):
        transformer_layer = CheckpointedParallelTransformerLayerPipe()
        post_transformer = PostTransformerBlock()
        model = SequentialWrapper(
            [transformer_layer, post_transformer],
            activation_checkpoint_interval=checkpoint_interval,
            activation_checkpoint_func=partial(checkpoint, use_reentrant=True),
            parent_class_name="GPT2ModelPipe",
        )

        hidden_states = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
        hidden_states.requires_grad_()
        context = (
            torch.tensor([0, 2, 4, 4, 4], dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.tensor(2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.bool),
        )

        output = model((hidden_states, *context))
        output.sum().backward()

        assert torch.is_tensor(output)
        assert len(post_transformer.received_context) == len(context)
        received_contexts = [
            post_transformer.received_context,
            *transformer_layer.received_contexts,
        ]
        for received_context in received_contexts:
            for actual, expected in zip(received_context, context):
                assert torch.equal(actual, expected)
                assert actual.dtype == expected.dtype
                assert actual.shape == expected.shape
        for tensor in context:
            assert not tensor.requires_grad
            assert tensor.grad is None

        expected_layer_calls = 2 if checkpoint_interval else 1
        assert len(transformer_layer.received_contexts) == expected_layer_calls

        return (
            output.detach(),
            hidden_states.grad,
            transformer_layer.weight.grad,
        )

    direct = run(checkpoint_interval=0)
    checkpointed = run(checkpoint_interval=1)

    for direct_tensor, checkpointed_tensor in zip(direct, checkpointed):
        assert torch.equal(direct_tensor, checkpointed_tensor)
