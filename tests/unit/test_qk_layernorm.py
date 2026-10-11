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

import copy
import socket
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import megatron.model.transformer as transformer
from megatron import mpu
from megatron.model.norms import NonParametricLayernorm, RMSNorm, ScaleNorm, get_norm
from megatron.model.utils import mark_norms_for_sequence_parallel_grad_sync
from megatron.neox_arguments import NeoXArgs
from megatron.neox_arguments.arguments import _validate_qk_norm_args
from megatron.model.transformer import (
    _apply_qk_norm_across_tp,
    _configure_qk_norm_tp_grads,
)

pytestmark = pytest.mark.cpu


def assert_close(actual, expected, rtol=1.3e-6, atol=1.0e-5):
    # torch.testing.assert_close needs torch>=1.9; CPU CI pins 1.8.
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (actual, expected)


def validate_qk_args(**overrides):
    args = {
        "use_qk_norm": True,
        "qk_norm": None,
        "qk_norm_type": "across_heads",
        "qk_norm_separate": False,
        "qk_norm_across_tp": False,
        "norm": "rmsnorm",
        "layernorm_fusion": False,
        "rmsnorm_fusion": False,
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_kv_heads": None,
        "model_parallel_size": 2,
    }
    args.update(overrides)
    _validate_qk_norm_args(**args)


@pytest.mark.parametrize(
    ("num_kv_heads", "qk_norm_separate"),
    [(None, False), (4, True), (2, True)],
)
def test_qk_norm_sharing_accepts_compatible_configuration(
    num_kv_heads, qk_norm_separate
):
    validate_qk_args(
        num_kv_heads=num_kv_heads,
        qk_norm_separate=qk_norm_separate,
    )


def test_qk_norm_sharing_rejects_different_sizes():
    with pytest.raises(ValueError) as exc_info:
        validate_qk_args(num_kv_heads=2, qk_norm_separate=False)

    message = str(exc_info.value)
    assert "normalized sizes differ (query=128, key=64)" in message
    assert "qk_norm_separate=True" in message
    assert "qk_norm_type='per_head'" in message
    assert "num_kv_heads" in message


def test_qk_norm_across_tp_validation_accepts_opt_in():
    validate_qk_args(qk_norm_across_tp=True)
    validate_qk_args(
        use_qk_norm=False,
        qk_norm_type="per_head",
        qk_norm_across_tp=False,
    )


@pytest.mark.parametrize(
    ("use_qk_norm", "qk_norm_type", "message"),
    [
        (False, "across_heads", "use_qk_norm=True"),
        (True, "per_head", "qk_norm_type='across_heads'"),
    ],
)
def test_qk_norm_across_tp_validation_rejects_incompatible_options(
    use_qk_norm, qk_norm_type, message
):
    with pytest.raises(ValueError, match=message):
        validate_qk_args(
            use_qk_norm=use_qk_norm,
            qk_norm_type=qk_norm_type,
            qk_norm_across_tp=True,
        )


def test_qk_norm_validation_rejects_unknown_type():
    with pytest.raises(ValueError, match="Invalid qk_norm_type"):
        validate_qk_args(qk_norm_type="unknown")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"norm": "te_rmsnorm"}, "Transformer Engine"),
        ({"qk_norm": "te_layernorm"}, "Transformer Engine"),
        ({"norm": "rmsnorm", "rmsnorm_fusion": True}, "rmsnorm_fusion=True"),
        (
            {"qk_norm": "layernorm", "layernorm_fusion": True},
            "layernorm_fusion=True",
        ),
    ],
)
def test_qk_norm_across_tp_rejects_internal_statistics_kernels(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_qk_args(qk_norm_across_tp=True, **overrides)


def test_explicit_qk_norm_allows_te_model_norm_with_cross_tp_stats():
    validate_qk_args(
        norm="te_rmsnorm",
        qk_norm="rmsnorm",
        qk_norm_across_tp=True,
    )


def test_get_norm_allows_qk_norm_to_override_model_norm():
    neox_args = SimpleNamespace(
        norm="rmsnorm",
        layernorm_epsilon=1.0e-5,
        layernorm_fusion=False,
    )

    norm, eps = get_norm(neox_args, norm_type="layernorm")

    assert norm is torch.nn.LayerNorm
    assert eps == neox_args.layernorm_epsilon


@pytest.mark.parametrize(
    "overrides",
    [
        {"qk_norm": "rmsnorm"},
        {"qk_norm_separate": True},
    ],
)
def test_qk_norm_options_require_qk_norm_to_be_enabled(overrides):
    with pytest.raises(ValueError, match="use_qk_norm=True"):
        validate_qk_args(use_qk_norm=False, **overrides)


def test_qk_norm_rejects_transformer_engine_attention():
    with pytest.raises(ValueError, match="te_mha"):
        validate_qk_args(te_mha=True)
    validate_qk_args(use_qk_norm=False, te_mha=True)


def test_renamed_use_qk_layernorm_is_rejected():
    with pytest.raises(ValueError, match="renamed to `use_qk_norm`"):
        NeoXArgs.from_dict({"use_qk_layernorm": True})


def _make_norm(norm_type, size, device="cpu"):
    norm_class = {
        "rmsnorm": RMSNorm,
        "layernorm": torch.nn.LayerNorm,
        "scalenorm": ScaleNorm,
    }[norm_type]
    norm = norm_class([size], eps=1.0e-6).to(device)
    with torch.no_grad():
        for parameter in norm.parameters():
            parameter.copy_(torch.linspace(0.75, 1.25, parameter.numel()))
    return norm


def _mark_tp_grads(q_norm, k_norm, qk_norm_type, norm_type, across_tp, world_size=2):
    # What model construction does, with sequence parallelism enabled.
    _configure_qk_norm_tp_grads(
        q_norm, k_norm, qk_norm_type, norm_type, across_tp, world_size
    )
    mark_norms_for_sequence_parallel_grad_sync(
        torch.nn.ModuleList([q_norm, k_norm]), SimpleNamespace(sequence_parallel=True)
    )


@pytest.mark.parametrize(
    ("qk_norm_type", "norm_type", "across_tp", "world_size", "hooked", "sliced"),
    [
        ("per_head", "rmsnorm", False, 2, True, False),
        ("across_heads", "rmsnorm", False, 2, False, True),
        ("across_heads", "rmsnorm", True, 2, False, True),
        ("across_heads", "scalenorm", True, 2, False, False),
        ("per_head", "rmsnorm", False, 1, False, False),
    ],
)
def test_qk_norm_tp_grad_handling(
    qk_norm_type, norm_type, across_tp, world_size, hooked, sliced
):
    norm = _make_norm(norm_type, 4)
    _mark_tp_grads(norm, norm, qk_norm_type, norm_type, across_tp, world_size)
    for parameter in norm.parameters():
        assert bool(parameter._backward_hooks) == hooked
        assert getattr(parameter, "model_parallel", False) == sliced


def test_qk_rmsnorm_across_tp_packs_stats_into_one_collective(monkeypatch):
    query = torch.tensor([[[1.0, 2.0]]])
    key = torch.tensor([[[2.0]]])
    q_norm = RMSNorm([2], eps=0.0)
    k_norm = RMSNorm([1], eps=0.0)
    with torch.no_grad():
        q_norm.scale.copy_(torch.tensor([2.0, 3.0]))
        k_norm.scale.copy_(torch.tensor([4.0]))
    model_parallel_group = object()
    calls = []

    def fake_all_reduce(packed_stats, op, group):
        calls.append((packed_stats.clone(), op, group))
        other_rank_stats = torch.tensor([[[25.0, 16.0]]])
        return packed_stats + other_rank_stats

    monkeypatch.setattr(transformer, "differentiable_all_reduce", fake_all_reduce)

    normalized_query, normalized_key = _apply_qk_norm_across_tp(
        query,
        key,
        q_norm,
        k_norm,
        norm_type="rmsnorm",
        model_parallel_world_size=2,
        model_parallel_group=model_parallel_group,
    )

    assert len(calls) == 1
    assert_close(calls[0][0], torch.tensor([[[5.0, 4.0]]]))
    assert calls[0][1] == torch.distributed.ReduceOp.SUM
    assert calls[0][2] is model_parallel_group
    assert_close(
        normalized_query,
        query * torch.rsqrt(torch.tensor(30.0 / 4.0)) * q_norm.scale,
    )
    assert_close(
        normalized_key,
        key * torch.rsqrt(torch.tensor(20.0 / 2.0)) * k_norm.scale,
    )


def test_qk_norm_layernorm_across_tp_packs_stats_into_one_collective(monkeypatch):
    query = torch.tensor([[[1.0, 3.0]]])
    key = torch.tensor([[[2.0]]])
    q_norm = torch.nn.LayerNorm([2], eps=0.0)
    k_norm = torch.nn.LayerNorm([1], eps=0.0)
    with torch.no_grad():
        q_norm.weight.copy_(torch.tensor([2.0, 3.0]))
        q_norm.bias.copy_(torch.tensor([1.0, -1.0]))
        k_norm.weight.copy_(torch.tensor([4.0]))
        k_norm.bias.copy_(torch.tensor([2.0]))
    model_parallel_group = object()
    calls = []

    def fake_all_reduce(packed_stats, op, group):
        calls.append((packed_stats.clone(), op, group))
        other_rank_stats = torch.tensor([[[12.0, 74.0, 6.0, 36.0]]])
        return packed_stats + other_rank_stats

    monkeypatch.setattr(transformer, "differentiable_all_reduce", fake_all_reduce)

    normalized_query, normalized_key = _apply_qk_norm_across_tp(
        query,
        key,
        q_norm,
        k_norm,
        norm_type="layernorm",
        model_parallel_world_size=2,
        model_parallel_group=model_parallel_group,
    )

    assert len(calls) == 1
    assert_close(calls[0][0], torch.tensor([[[4.0, 10.0, 2.0, 4.0]]]))
    assert calls[0][1] == torch.distributed.ReduceOp.SUM
    assert calls[0][2] is model_parallel_group
    assert_close(
        normalized_query,
        (query - 4.0) * torch.rsqrt(torch.tensor(5.0)) * q_norm.weight + q_norm.bias,
    )
    assert_close(
        normalized_key,
        (key - 4.0) * torch.rsqrt(torch.tensor(4.0)) * k_norm.weight + k_norm.bias,
    )


def test_non_parametric_layernorm_across_tp_has_no_affine(monkeypatch):
    query = torch.tensor([[[1.0, 3.0]]])
    key = torch.tensor([[[2.0]]])
    q_norm = NonParametricLayernorm([2], eps=0.0)
    k_norm = NonParametricLayernorm([1], eps=0.0)

    def fake_all_reduce(packed_stats, op, group):
        return packed_stats + torch.tensor([[[12.0, 74.0, 6.0, 36.0]]])

    monkeypatch.setattr(transformer, "differentiable_all_reduce", fake_all_reduce)

    normalized_query, normalized_key = _apply_qk_norm_across_tp(
        query,
        key,
        q_norm,
        k_norm,
        norm_type="non_parametric_layernorm",
        model_parallel_world_size=2,
        model_parallel_group=object(),
    )

    assert_close(normalized_query, (query - 4.0) * torch.rsqrt(torch.tensor(5.0)))
    assert_close(normalized_key, (key - 4.0) * torch.rsqrt(torch.tensor(4.0)))


def test_qk_scalenorm_across_tp_packs_stats_into_one_collective(monkeypatch):
    query = torch.tensor([[[1.0, 2.0]]])
    key = torch.tensor([[[2.0]]])
    q_norm = ScaleNorm([2], eps=0.0)
    k_norm = ScaleNorm([1], eps=0.0)
    with torch.no_grad():
        q_norm.g.fill_(2.0)
        k_norm.g.fill_(4.0)
    calls = []

    def fake_all_reduce(packed_stats, op, group):
        calls.append(packed_stats.clone())
        return packed_stats + torch.tensor([[[25.0, 16.0]]])

    monkeypatch.setattr(transformer, "differentiable_all_reduce", fake_all_reduce)

    normalized_query, normalized_key = _apply_qk_norm_across_tp(
        query,
        key,
        q_norm,
        k_norm,
        norm_type="scalenorm",
        model_parallel_world_size=2,
        model_parallel_group=object(),
    )

    assert len(calls) == 1
    assert_close(calls[0], torch.tensor([[[5.0, 4.0]]]))
    assert_close(normalized_query, query / torch.sqrt(torch.tensor(30.0)) * 2.0)
    assert_close(normalized_key, key / torch.sqrt(torch.tensor(20.0)) * 4.0)


def _make_norm_with_matching_shard(norm_type, local_size, rank, device):
    full_size = local_size * 2
    eps = 1.0e-6
    if norm_type == "rmsnorm":
        local_norm = RMSNorm([local_size], eps=eps).to(device)
        full_norm = RMSNorm([full_size], eps=eps).to(device)
        full_scale = torch.linspace(0.75, 1.25, full_size, device=device)
        with torch.no_grad():
            full_norm.scale.copy_(full_scale)
            local_norm.scale.copy_(full_scale.chunk(2)[rank])
    elif norm_type == "layernorm":
        local_norm = torch.nn.LayerNorm([local_size], eps=eps).to(device)
        full_norm = torch.nn.LayerNorm([full_size], eps=eps).to(device)
        full_weight = torch.linspace(0.75, 1.25, full_size, device=device)
        full_bias = torch.linspace(-0.2, 0.2, full_size, device=device)
        with torch.no_grad():
            full_norm.weight.copy_(full_weight)
            full_norm.bias.copy_(full_bias)
            local_norm.weight.copy_(full_weight.chunk(2)[rank])
            local_norm.bias.copy_(full_bias.chunk(2)[rank])
    elif norm_type == "non_parametric_layernorm":
        local_norm = NonParametricLayernorm([local_size], eps=eps).to(device)
        full_norm = NonParametricLayernorm([full_size], eps=eps).to(device)
    else:
        local_norm = ScaleNorm([local_size], eps=eps).to(device)
        full_norm = ScaleNorm([full_size], eps=eps).to(device)
        with torch.no_grad():
            local_norm.g.fill_(1.25)
            full_norm.g.fill_(1.25)
    return local_norm, full_norm


def _assert_norm_parameter_grads_match_full_projection(
    norm_type, local_norm, full_norm, rank
):
    for (_, local_parameter), (_, full_parameter) in zip(
        local_norm.named_parameters(), full_norm.named_parameters()
    ):
        if norm_type == "scalenorm":
            expected_grad = full_parameter.grad
        else:
            expected_grad = full_parameter.grad.chunk(2)[rank]
        assert_close(local_parameter.grad, expected_grad)


@contextmanager
def _tp_rank(rank, world_size, port):
    # NCCL on GPUs when available, otherwise gloo on CPU so this runs in CPU CI.
    use_cuda = torch.cuda.device_count() >= world_size
    if use_cuda:
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    mpu.initialize_model_parallel(world_size)
    try:
        yield torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _spawn_two_ranks(worker):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as test_socket:
        test_socket.bind(("127.0.0.1", 0))
        port = test_socket.getsockname()[1]
    mp.spawn(worker, args=(2, port), nprocs=2, join=True)


def _run_distributed_qk_norm_test(rank, world_size, port):
    with _tp_rank(rank, world_size, port) as device:
        full_query_values = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], device=device)
        full_key_values = torch.tensor([[[3.0, 6.0]]], device=device)
        full_query_coefficients = torch.tensor(
            [[[0.5, -1.0, 1.5, -0.25]]], device=device
        )
        full_key_coefficients = torch.tensor([[[0.75, -1.25]]], device=device)

        for norm_type in (
            "rmsnorm",
            "layernorm",
            "non_parametric_layernorm",
            "scalenorm",
        ):
            local_query = (
                full_query_values.chunk(world_size, dim=-1)[rank]
                .clone()
                .requires_grad_(True)
            )
            local_key = (
                full_key_values.chunk(world_size, dim=-1)[rank]
                .clone()
                .requires_grad_(True)
            )
            full_query = full_query_values.clone().requires_grad_(True)
            full_key = full_key_values.clone().requires_grad_(True)
            local_q_norm, full_q_norm = _make_norm_with_matching_shard(
                norm_type, local_query.size(-1), rank, device
            )
            local_k_norm, full_k_norm = _make_norm_with_matching_shard(
                norm_type, local_key.size(-1), rank, device
            )
            _mark_tp_grads(local_q_norm, local_k_norm, "across_heads", norm_type, True)

            local_query_output, local_key_output = _apply_qk_norm_across_tp(
                local_query,
                local_key,
                local_q_norm,
                local_k_norm,
                norm_type=norm_type,
                model_parallel_world_size=world_size,
                model_parallel_group=mpu.get_model_parallel_group(),
            )
            full_query_output = full_q_norm(full_query)
            full_key_output = full_k_norm(full_key)
            expected_query_output = full_query_output.chunk(world_size, dim=-1)[rank]
            expected_key_output = full_key_output.chunk(world_size, dim=-1)[rank]
            assert_close(
                local_query_output, expected_query_output, rtol=1.0e-5, atol=1.0e-6
            )
            assert_close(
                local_key_output, expected_key_output, rtol=1.0e-5, atol=1.0e-6
            )

            local_loss = (
                local_query_output
                * full_query_coefficients.chunk(world_size, dim=-1)[rank]
            ).sum() + (
                local_key_output * full_key_coefficients.chunk(world_size, dim=-1)[rank]
            ).sum()
            full_loss = (full_query_output * full_query_coefficients).sum() + (
                full_key_output * full_key_coefficients
            ).sum()
            local_loss.backward()
            full_loss.backward()

            assert_close(
                local_query.grad,
                full_query.grad.chunk(world_size, dim=-1)[rank],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            assert_close(
                local_key.grad,
                full_key.grad.chunk(world_size, dim=-1)[rank],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            _assert_norm_parameter_grads_match_full_projection(
                norm_type, local_q_norm, full_q_norm, rank
            )
            _assert_norm_parameter_grads_match_full_projection(
                norm_type, local_k_norm, full_k_norm, rank
            )


def test_qk_norm_across_tp_matches_full_projection_forward_and_backward():
    _spawn_two_ranks(_run_distributed_qk_norm_test)


def _run_distributed_qk_norm_grad_sync_test(rank, world_size, port):
    with _tp_rank(rank, world_size, port) as device:
        # [heads, head_dim]; each rank holds one head.
        query = torch.tensor([[1.0, 2.0], [4.0, 8.0]], device=device)
        key = torch.tensor([[3.0, 6.0], [5.0, -2.0]], device=device)
        query_coefficients = torch.tensor([[0.5, -1.0], [1.5, -0.25]], device=device)
        key_coefficients = torch.tensor([[0.75, -1.25], [-0.5, 2.0]], device=device)
        local = slice(rank, rank + 1)

        def loss(norm, heads):
            return (norm(query[heads]) * query_coefficients[heads]).sum() + (
                norm(key[heads]) * key_coefficients[heads]
            ).sum()

        for norm_type in ("rmsnorm", "layernorm", "scalenorm"):
            # per_head: one shared [H] weight, so every rank gets the TP=1 grad.
            norm = _make_norm(norm_type, 2, device)
            reference = copy.deepcopy(norm)
            _mark_tp_grads(norm, norm, "per_head", norm_type, False)
            loss(norm, local).backward()
            loss(reference, slice(None)).backward()
            for parameter, expected in zip(norm.parameters(), reference.parameters()):
                assert_close(parameter.grad, expected.grad)

            # Rank-local across_heads: weights are private, so grads stay local.
            norm = _make_norm(norm_type, 2, device)
            reference = copy.deepcopy(norm)
            _mark_tp_grads(norm, norm, "across_heads", norm_type, False)
            loss(norm, local).backward()
            loss(reference, local).backward()
            for parameter, expected in zip(norm.parameters(), reference.parameters()):
                assert_close(parameter.grad, expected.grad)


def test_qk_norm_grads_are_synced_per_mode_across_tp():
    _spawn_two_ranks(_run_distributed_qk_norm_grad_sync_test)
