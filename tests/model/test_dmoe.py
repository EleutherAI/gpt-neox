# Copyright (c) 2025, EleutherAI
# Licensed under the Apache 2.0 licence.

"""
▸ Part 1 – expert‑token helper utilities:
    * `get_expert_tokens_for_rank`
    * `get_expert_token_counts_for_rank`

▸ Part 2 – lightweight router (`TopKTokenChoiceRouter`)
    * shape & range of returned weights / indices
    * determinism under identical input

"""

import types
import torch
import pytest
import importlib


@pytest.fixture(autouse=True)
def patch_mpu(monkeypatch):
    """
    Pretend we have a 2‑way tensor‑parallel group; most MoE helpers only query
    `get_model_parallel_world_size` and `get_model_parallel_rank`.
    """
    import megatron.mpu as mpu

    monkeypatch.setattr(mpu, "get_model_parallel_world_size", lambda: 2, raising=False)
    # `rank` will be injected per‑test case
    yield


def _set_rank(monkeypatch, rank: int):
    import megatron.mpu as mpu
    monkeypatch.setattr(mpu, "get_model_parallel_rank", lambda: rank, raising=False)


# Part 1 – expert‑token split / gather helpers
@pytest.mark.parametrize("rank", [0, 1])
def test_expert_token_helpers(monkeypatch, rank):
    """
    A tiny batch of 6 routed tokens divided among 4 experts with the pattern
    [2,1,0,3].  With world_size==2 each rank owns 2 experts ⇒ verify that
     the expected slices/counts are returned.
    """
    from megatron.mpu.initialize import (
        get_expert_tokens_for_rank,
        get_expert_token_counts_for_rank,
    )

    _set_rank(monkeypatch, rank)

    tokens_per_expert = torch.tensor([2, 1, 0, 3])          # len == num_experts
    routed          = torch.arange(6*3).view(6, 3)          # shape (6, 3)

    # ‑‑ expected slice for this fake rank
    # cumulative sums → [2,3,3,6]; rank 0 gets experts 0&1, rank 1 gets 2&3
    start = 0 if rank == 0 else 3
    end   = 3 if rank == 0 else 6
    want_slice = routed[start:end]

    out_tokens = get_expert_tokens_for_rank(routed, tokens_per_expert)
    out_counts = get_expert_token_counts_for_rank(tokens_per_expert)

    assert torch.equal(out_tokens, want_slice)
    assert out_counts.tolist() == ([2, 1] if rank == 0 else [0, 3])


# Part 2 – Top‑K token‑choice router
def _dummy_args(num_experts=8, top_k=2, hidden_size=16):
    """Return a minimal object that TopKTokenChoiceRouter expects."""
    return types.SimpleNamespace(
        hidden_size      = hidden_size,
        moe_num_experts  = num_experts,
        moe_top_k        = top_k,
        moe_jitter_eps   = None,
        params_dtype     = torch.float32,    # keep everything on CPU
    )


@pytest.mark.parametrize("top_k", [1, 2])
def test_router_shapes_and_range(top_k):
    """Router must return (batch, top_k) tensors; indices < num_experts."""
    mod = importlib.import_module("megatron.model.router")
    Router = mod.TopKTokenChoiceRouter

    args  = _dummy_args(num_experts=5, top_k=top_k, hidden_size=32)
    router = Router(args, init_method=torch.nn.init.uniform_)

    seq, bs = 4, 3
    x = torch.randn(seq, bs, args.hidden_size)

    w, idx = router(x)

    assert w.shape == (seq * bs, top_k)
    assert idx.shape == (seq * bs, top_k)
    assert torch.all(idx < args.moe_num_experts)
    # Probabilities must be positive and ≤1
    assert torch.all(w >= 0) and torch.all(w <= 1)

    # Deterministic behaviour for identical input (no jitter, eval mode).
    router.eval()
    w2, idx2 = router(x)
    assert torch.equal(w, w2)
    assert torch.equal(idx, idx2)

