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

import sys
from types import SimpleNamespace

import pytest
import torch

from megatron import mpu
from megatron.model.gpt2_model import cross_entropy


pytestmark = pytest.mark.cpu


def test_cross_entropy_routes_to_fused_implementation(monkeypatch):
    output = torch.randn(2, 3, 5)
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss_mask = torch.ones(2, 3)
    tp_group = object()
    calls = []

    def fused_vocab_parallel_cross_entropy(logits, targets, group):
        calls.append((logits, targets, group))
        return torch.full_like(targets, 2.0, dtype=output.dtype)

    monkeypatch.setitem(
        sys.modules,
        "megatron.fused_kernels.fused_cross_entropy",
        SimpleNamespace(
            fused_vocab_parallel_cross_entropy=fused_vocab_parallel_cross_entropy
        ),
    )
    monkeypatch.setattr(mpu, "get_model_parallel_group", lambda: tp_group)

    loss = cross_entropy(output, (labels, loss_mask), _fused=True)

    assert loss.item() == 2.0
    assert calls == [(output, labels, tp_group)]


def test_fused_cross_entropy_preserves_fp16_loss_dtype(monkeypatch):
    output = torch.randn(2, 3, 5, dtype=torch.float16)
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss_mask = torch.ones(2, 3, dtype=torch.float16)

    def fused_vocab_parallel_cross_entropy(logits, targets, group):
        return torch.full_like(targets, 2.0, dtype=torch.float32)

    monkeypatch.setitem(
        sys.modules,
        "megatron.fused_kernels.fused_cross_entropy",
        SimpleNamespace(
            fused_vocab_parallel_cross_entropy=fused_vocab_parallel_cross_entropy
        ),
    )
    monkeypatch.setattr(mpu, "get_model_parallel_group", object)

    loss = cross_entropy(output, (labels, loss_mask), _fp16=True, _fused=True)

    assert loss.dtype == torch.float16
    assert loss.item() == 2.0


def test_fused_cross_entropy_matches_reference_and_uses_two_collectives(monkeypatch):
    from megatron.fused_kernels.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )

    fused_logits = torch.randn(2, 3, 7, requires_grad=True)
    reference_logits = fused_logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, fused_logits.size(-1), (2, 3))
    collective_ops = []

    def all_reduce(tensor, op, group):
        collective_ops.append((tensor.numel(), op, group))

    tp_group = object()
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    fused_loss = fused_vocab_parallel_cross_entropy(fused_logits, targets, tp_group)
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.view(-1, reference_logits.size(-1)),
        targets.view(-1),
        reduction="none",
    ).view_as(targets)

    torch.testing.assert_close(fused_loss, reference_loss)
    fused_loss.sum().backward()
    reference_loss.sum().backward()
    torch.testing.assert_close(fused_logits.grad, reference_logits.grad)

    assert collective_ops == [
        (targets.numel(), torch.distributed.ReduceOp.MAX, tp_group),
        (2 * targets.numel(), torch.distributed.ReduceOp.SUM, tp_group),
    ]


def test_fused_cross_entropy_restores_half_gradient_dtype(monkeypatch):
    from megatron.fused_kernels.fused_cross_entropy import (
        fused_vocab_parallel_cross_entropy,
    )

    logits = torch.randn(2, 3, 7, dtype=torch.float16, requires_grad=True)
    targets = torch.randint(0, logits.size(-1), (2, 3))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)

    loss = fused_vocab_parallel_cross_entropy(logits, targets, object())
    loss.sum().backward()

    assert loss.dtype == torch.float32
    assert logits.grad.dtype == torch.float16
