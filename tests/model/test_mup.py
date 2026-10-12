# Copyright (c) 2025, EleutherAI
# Licensed under the Apache 2.0 license.

import types
import torch
import pytest

from megatron.model.utils import get_params_for_weight_decay_optimization
from megatron.learning_rates import AnnealingLR


class TinyNet(torch.nn.Module):
    """Just enough structure to exercise the param‑group builder."""
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)           # should get weight‑decay
        self.norm = torch.nn.LayerNorm(4)          # should be no‑decay


@pytest.fixture(scope="module")
def dummy_args():
    # Only the attributes that `get_params_for_weight_decay_optimization`
    # actually accesses.
    return types.SimpleNamespace(weight_decay=0.1)


def _new_scheduler(optimizer, use_mup, width_mult):
    """
    Construct an AnnealingLR and monkey‑patch ``get_lr`` so the test is
    independent of the exact schedule math.
    """
    sched = AnnealingLR(
        optimizer,
        start_lr=0.0,
        max_lr=0.02,
        min_lr=0.0,
        warmup_iter=0,
        total_iters=1,
        decay_style="constant",
        use_checkpoint_lr_scheduler=False,
        override_lr_scheduler=False,
        use_mup=use_mup,
        mup_width_multiplier=width_mult,
    )
    # Force the scheduler to think LR should be 0.02 every step
    AnnealingLR.get_lr = lambda self: 0.02
    return sched


def test_param_groups_have_lr_adjust(dummy_args):
    """Builder should tag both WD and no‑WD groups with ``lr_adjust``."""
    net = TinyNet()
    groups = get_params_for_weight_decay_optimization(net, dummy_args)

    assert len(groups) == 2
    assert all(g.get("lr_adjust", False) for g in groups), (
        "Every param‑group returned by the builder must carry lr_adjust=True "
        "so muP knows to divide its LR."
    )


@pytest.mark.parametrize("use_mup,expected_factor", [(True, 4.0), (False, 1.0)])
def test_scheduler_scales_learning_rate(monkeypatch, dummy_args, use_mup, expected_factor):
    """
    When `use_mup` is True the LR of *lr_adjust* groups must be divided by
    ``mup_width_multiplier``; otherwise, it must stay unchanged.
    """
    net = TinyNet()
    param_groups = get_params_for_weight_decay_optimization(net, dummy_args)

    optimizer = torch.optim.SGD(param_groups, lr=0.0) # fine for sanity checking
    width_mult = 4.0
    sched = _new_scheduler(optimizer, use_mup=use_mup, width_mult=width_mult)

    sched.step()

    lrs = [g["lr"] for g in optimizer.param_groups]
    assert pytest.approx(lrs[0], rel=1e-7) == 0.02 / expected_factor
    assert pytest.approx(lrs[1], rel=1e-7) == 0.02 / expected_factor

