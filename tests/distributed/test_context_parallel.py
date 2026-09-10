# Copyright (c) 2025, EleutherAI
# Licensed under the Apache 2.0 license.

"""
Unit‑tests for context‑parallelism

We patch `megatron.mpu.get_context_parallel_*` so that we don't have to set up distributed on the gh CI runner
2‑way context‑parallel world is running, then verify that:

1. `zigzag_data` returns the correct slice for each (fake) rank.
2. `RotaryEmbedding` builds `cos_cached` / `sin_cached` using the same
   zig‑zag time‑indices.

"""

import torch
import pytest
import megatron.mpu as mpu
from megatron.mpu.data import zigzag_data
from megatron.model.positional_embeddings import RotaryEmbedding


@pytest.mark.parametrize("rank", [0, 1])
def test_zigzag_and_rotary(monkeypatch, rank):
    """
    Simulate a 2‑GPU context‑parallel group and check that both the low‑level
    zig‑zag utility and the higher‑level rotary‑embedding cache behave as
    expected on each rank.
    """
    # Patch the MPU helpers to fake a 2‑way group
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)

    # zigzag_data
    seq_dim = 1
    x = torch.arange(16).view(2, 8)        # shape: (batch=2, seq=8)

    # Compute the expected zig‑zag slice manually
    chunks = torch.chunk(x, 2 * 2, dim=seq_dim)   # 4 chunks of length 2
    expected = (
        torch.cat((chunks[0], chunks[-1]), dim=seq_dim)
        if rank == 0
        else torch.cat((chunks[1], chunks[-2]), dim=seq_dim)
    )

    out = zigzag_data(x, seq_dim=seq_dim)
    assert torch.equal(out, expected), "zig‑zag sharding mismatch"

    # RotaryEmbedding cache
    dim = 8
    rope = RotaryEmbedding(
        dim=dim,
        max_seq_len=8,
        base=10_000,
        precision=torch.float32,
        zigzag=True,
    )

    # Re‑create the ‘t’ indices that _prepare_cache() should have used
    full_t = torch.arange(8)
    expected_t = (
        torch.cat((full_t[:2], full_t[-2:]))     # rank 0
        if rank == 0
        else torch.cat((full_t[2:4], full_t[-4:-2]))  # rank 1
    )

    inv_freq = 1.0 / (10_000 ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.einsum("i,j->ij", expected_t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_ref, sin_ref = emb.cos(), emb.sin()

    assert rope.cos_cached.shape == cos_ref.shape
    assert rope.sin_cached.shape == sin_ref.shape
    assert torch.allclose(rope.cos_cached, cos_ref, atol=1e-6)
    assert torch.allclose(rope.sin_cached, sin_ref, atol=1e-6)

