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


def cosine_scheduler(warmup_iter):
    optimizer = SimpleNamespace(param_groups=[{}])
    return AnnealingLR(
        optimizer,
        start_lr=1.0,
        warmup_iter=warmup_iter,
        total_iters=100,
        decay_style="cosine",
        last_iter=0,
        min_lr=0.1,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("warmup_iter", [0, 10])
def test_cosine_lr_stays_at_min_lr_after_decay_iters(warmup_iter):
    # lr_decay_iters / lr_decay_fraction can make the decay window shorter
    # than training, so the scheduler keeps stepping past total_iters.
    scheduler = cosine_scheduler(warmup_iter)
    scheduler.step(100)
    assert scheduler.get_lr() == pytest.approx(0.1)
    for step in (101, 150, 190, 250):
        scheduler.step(step)
        assert scheduler.get_lr() == pytest.approx(0.1)
        assert scheduler.optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
