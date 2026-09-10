# Copyright (c) 2025, EleutherAI
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

"""Unit tests for the AnnealingLR learning rate scheduler.

Tests cover all decay styles including the WSD (Warmup-Stable-Decay) schedule.
These tests are CPU-only and do not require CUDA or distributed training.

Contact: @Rakshitha-Ireddi (GitHub)
"""

import math
import importlib
import types
import pytest
import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap: create a lightweight mock of the 'megatron' package so that
# learning_rates.py can be imported without needing deepspeed / CUDA.
# ---------------------------------------------------------------------------

# Build a minimal megatron stub that provides print_rank_0
_megatron_stub = types.ModuleType("megatron")
_megatron_stub.print_rank_0 = lambda *a, **kw: None
sys.modules["megatron"] = _megatron_stub

# Now load learning_rates from its file path
_lr_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "megatron", "learning_rates.py")
)
_spec = importlib.util.spec_from_file_location("megatron.learning_rates", _lr_path)
_lr_mod = importlib.util.module_from_spec(_spec)
sys.modules["megatron.learning_rates"] = _lr_mod
_spec.loader.exec_module(_lr_mod)

AnnealingLR = _lr_mod.AnnealingLR

import torch


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_optimizer(lr=1.0):
    """Create a trivial optimizer for scheduler tests."""
    param = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


# ---------------------------------------------------------------------------
# Tests for existing decay styles (regression)
# ---------------------------------------------------------------------------

class TestConstantDecay:
    def test_constant_lr(self):
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=0, total_iters=100,
            decay_style="constant", last_iter=0,
        )
        for i in range(1, 101):
            sched.step(i)
            assert sched.get_lr() == pytest.approx(1e-3)


class TestLinearDecay:
    def test_linear_reaches_zero(self):
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1.0, warmup_iter=0, total_iters=100,
            decay_style="linear", last_iter=0,
        )
        sched.step(100)
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-9)


class TestCosineDecay:
    def test_cosine_midpoint(self):
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1.0, warmup_iter=0, total_iters=100,
            decay_style="cosine", last_iter=0, min_lr=0.0,
        )
        sched.step(50)
        # At midpoint, cosine decay should be at 0.5 * (cos(pi*0.5) + 1) = 0.5
        assert sched.get_lr() == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests for WSD decay style
# ---------------------------------------------------------------------------

class TestWSDDecay:
    """Tests for the Warmup-Stable-Decay learning rate schedule."""

    def test_warmup_phase(self):
        """During warmup, LR should increase linearly from 0 to start_lr."""
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=100, total_iters=1000,
            decay_style="wsd", last_iter=0, min_lr=1e-5,
            wsd_decay_ratio=0.1,
        )
        # At step 50, LR should be 50/100 * 1e-3
        sched.step(50)
        assert sched.get_lr() == pytest.approx(0.5e-3, rel=1e-6)

        # At step 100 (end of warmup), LR should be start_lr
        sched.step(100)
        assert sched.get_lr() == pytest.approx(1e-3, rel=1e-6)

    def test_stable_phase(self):
        """After warmup and before decay, LR should be constant at start_lr."""
        opt = _make_optimizer()
        total_iters = 1000
        warmup_iter = 100
        wsd_decay_ratio = 0.1
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=warmup_iter,
            total_iters=total_iters, decay_style="wsd", last_iter=0,
            min_lr=1e-5, wsd_decay_ratio=wsd_decay_ratio,
        )
        # Post-warmup iters = 900.  Decay iters = 90.  Stable iters = 810.
        # So stable phase spans steps 101 through 910.
        for step in [150, 500, 800, 910]:
            sched.step(step)
            assert sched.get_lr() == pytest.approx(1e-3, rel=1e-6), (
                f"Expected start_lr during stable phase at step {step}"
            )

    def test_decay_phase(self):
        """During the final decay phase, LR should cosine-anneal to min_lr."""
        opt = _make_optimizer()
        total_iters = 1000
        warmup_iter = 100
        wsd_decay_ratio = 0.1
        start_lr = 1e-3
        min_lr = 1e-5
        sched = AnnealingLR(
            opt, start_lr=start_lr, warmup_iter=warmup_iter,
            total_iters=total_iters, decay_style="wsd", last_iter=0,
            min_lr=min_lr, wsd_decay_ratio=wsd_decay_ratio,
        )
        # Post-warmup = 900, decay_iters = 90, stable_iters = 810
        # Decay starts at step 100 + 810 = 910
        # At end of training (step 1000), LR should be ~min_lr
        sched.step(1000)
        assert sched.get_lr() == pytest.approx(min_lr, abs=1e-7)

        # At the midpoint of decay (step 955 = 910 + 45)
        sched.step(955)
        expected = min_lr + (start_lr - min_lr) / 2.0 * (
            math.cos(math.pi * 45 / 90) + 1
        )
        assert sched.get_lr() == pytest.approx(expected, rel=1e-5)

    def test_decay_start_equals_start_lr(self):
        """At the first iteration of the decay phase, LR should still be ~start_lr."""
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=100, total_iters=1000,
            decay_style="wsd", last_iter=0, min_lr=0.0,
            wsd_decay_ratio=0.1,
        )
        # Decay starts at step 911 (first step after stable)
        sched.step(911)
        # Should be very close to start_lr (one step into cosine)
        assert sched.get_lr() > 0.99e-3

    def test_state_dict_round_trip(self):
        """state_dict / load_state_dict should preserve wsd_decay_ratio."""
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=10, total_iters=100,
            decay_style="wsd", last_iter=50, min_lr=1e-5,
            wsd_decay_ratio=0.2,
        )
        sd = sched.state_dict()
        assert "wsd_decay_ratio" in sd
        assert sd["wsd_decay_ratio"] == 0.2

        # Create a new scheduler and load the state dict
        opt2 = _make_optimizer()
        sched2 = AnnealingLR(
            opt2, start_lr=1e-3, warmup_iter=10, total_iters=100,
            decay_style="wsd", last_iter=0, min_lr=1e-5,
            wsd_decay_ratio=0.2,
        )
        sched2.load_state_dict(sd)
        assert sched2.wsd_decay_ratio == 0.2
        assert sched2.num_iters == 50

    def test_no_warmup_wsd(self):
        """WSD with zero warmup should go straight to stable then decay."""
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=0, total_iters=100,
            decay_style="wsd", last_iter=0, min_lr=0.0,
            wsd_decay_ratio=0.2,
        )
        # Stable phase: steps 1-80
        sched.step(1)
        assert sched.get_lr() == pytest.approx(1e-3)
        sched.step(80)
        assert sched.get_lr() == pytest.approx(1e-3)

        # End of decay: step 100
        sched.step(100)
        assert sched.get_lr() == pytest.approx(0.0, abs=1e-9)

    def test_full_decay_wsd(self):
        """WSD with wsd_decay_ratio=1.0 should behave like cosine decay after warmup."""
        opt = _make_optimizer()
        sched = AnnealingLR(
            opt, start_lr=1e-3, warmup_iter=0, total_iters=100,
            decay_style="wsd", last_iter=0, min_lr=0.0,
            wsd_decay_ratio=1.0,
        )
        # With ratio=1.0, there's no stable phase, so it's all cosine decay
        sched.step(50)
        expected = (1e-3 / 2.0) * (math.cos(math.pi * 50 / 100) + 1)
        assert sched.get_lr() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Test warmup is shared across all styles
# ---------------------------------------------------------------------------

class TestWarmup:
    @pytest.mark.parametrize("style", ["constant", "linear", "cosine", "exponential", "wsd"])
    def test_warmup_common(self, style):
        """All decay styles should have the same linear warmup behavior."""
        opt = _make_optimizer()
        kwargs = {}
        if style == "wsd":
            kwargs["wsd_decay_ratio"] = 0.1
        sched = AnnealingLR(
            opt, start_lr=1.0, warmup_iter=10, total_iters=100,
            decay_style=style, last_iter=0, **kwargs,
        )
        sched.step(5)
        assert sched.get_lr() == pytest.approx(0.5, rel=1e-6)
