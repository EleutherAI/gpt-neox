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

from megatron.learning_rates import AnnealingLR


def build_scheduler(warmup_iter, decay_style="cosine"):
    optimizer = SimpleNamespace(param_groups=[{}])
    return AnnealingLR(
        optimizer,
        start_lr=1.0,
        warmup_iter=warmup_iter,
        total_iters=100,
        decay_style=decay_style,
        last_iter=0,
        min_lr=0.1,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("warmup_iter", [0, 10])
def test_cosine_lr_stays_at_min_lr_after_decay_iters(warmup_iter):
    # lr_decay_iters / lr_decay_fraction can make the decay window shorter
    # than training, so the scheduler keeps stepping past total_iters.
    scheduler = build_scheduler(warmup_iter)
    scheduler.step(100)
    assert scheduler.get_lr() == pytest.approx(0.1)
    for step in (101, 150, 190, 250):
        scheduler.step(step)
        assert scheduler.get_lr() == pytest.approx(0.1)
        assert scheduler.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


@pytest.mark.cpu
@pytest.mark.parametrize("warmup_iter", [0, 10])
def test_cosine_lr_unchanged_inside_decay_window(warmup_iter):
    scheduler = build_scheduler(warmup_iter)
    midpoint = warmup_iter + (100 - warmup_iter) // 2
    for step, expected in ((warmup_iter + 1, 1.0), (midpoint, 0.55), (100, 0.1)):
        scheduler.step(step)
        assert scheduler.get_lr() == pytest.approx(expected, abs=1e-3)


@pytest.mark.cpu
@pytest.mark.parametrize("warmup_iter", [0, 10])
@pytest.mark.parametrize("decay_style", ["constant", "linear", "cosine", "exponential"])
def test_lr_never_increases_after_warmup(decay_style, warmup_iter):
    scheduler = build_scheduler(warmup_iter, decay_style)
    scheduler.step(warmup_iter)
    prev = scheduler.get_lr()
    # step() with no argument is how DeepSpeed drives the scheduler
    for _ in range(300):
        scheduler.step()
        lr = scheduler.get_lr()
        assert 0.1 <= lr <= prev
        prev = lr
