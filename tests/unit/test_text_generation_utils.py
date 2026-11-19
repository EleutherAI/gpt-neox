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
from unittest.mock import Mock

import pytest
import torch

from megatron import text_generation_utils as generation
from megatron.model.utils import SequentialWrapper


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="stream_tokens uses CUDA"
)


@pytest.fixture
def args(monkeypatch):
    # A single rank needs no communication.
    monkeypatch.setattr(generation.mpu, "get_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(generation.mpu, "get_model_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    return SimpleNamespace(
        tokenizer=SimpleNamespace(eod=0),
        seq_length=16,
        padded_vocab_size=2,
        eod_mask_loss=False,
        is_pipe_parallel=False,
        return_logits=True,
    )


@pytest.mark.parametrize("prompt", [[1], [1, 1]], ids=["one-token", "two-token"])
@pytest.mark.parametrize("recompute", [False, True], ids=["cached", "recompute"])
def test_return_logits_starts_at_first_decode_step(args, prompt, recompute):
    def position_logits(inputs):
        positions = inputs[1].float()
        return torch.stack((positions, positions + 1), dim=-1)  # Always pick token 1.

    maximum_tokens = 8
    steps = list(
        generation.stream_tokens(
            args,
            Mock(module=Mock(side_effect=position_logits)),
            [prompt],
            maximum_tokens=maximum_tokens,
            recompute=recompute,
        )
    )
    assert len(steps) == maximum_tokens
    logits = steps[-1][3]
    first_position = len(prompt) - 1
    expected = [
        [p, p + 1] for p in range(first_position, first_position + maximum_tokens)
    ]
    torch.testing.assert_close(logits, logits.new_tensor(expected))


class TokenSum(torch.nn.Module):
    """A stateful layer with deterministic logits that reveal any leaked or lost prefix."""

    def __init__(self):
        super().__init__()
        self.layer_past = 100  # Simulate a model previously used by another caller.

    def forward(self, inputs):
        sums = inputs[0].float().cumsum(dim=1)
        if self.layer_past is not None:
            sums += self.layer_past
        self.layer_past = sums[:, -1:]
        return torch.stack((sums, sums + 1), dim=-1)


@pytest.fixture
def cached_model():
    layer = TokenSum()
    module = SequentialWrapper(
        [layer], activation_checkpoint_interval=0, activation_checkpoint_func=None
    )
    return Mock(module=module), layer


@pytest.mark.parametrize("recompute", [False, True], ids=["cached", "recompute"])
def test_cache_is_scoped_to_one_prompt(args, cached_model, recompute):
    model, layer = cached_model
    for prompt in [[1, 1], [1]]:
        *_, last = generation.stream_tokens(
            args, model, [prompt], maximum_tokens=3, recompute=recompute
        )
        scores = last[3][:, 0]
        expected = [sum(prompt) + i for i in range(3)]
        torch.testing.assert_close(scores, scores.new_tensor(expected))
        assert layer.layer_past is None


@pytest.mark.parametrize("stop", ["eos", "close", "error"])
def test_stream_releases_cache_on_early_exit(args, cached_model, monkeypatch, stop):
    model, layer = cached_model
    stream = generation.stream_tokens(
        args, model, [[1]], maximum_tokens=4, eos_token_id=1 if stop == "eos" else 0
    )
    next(stream)
    assert layer.layer_past is not None
    if stop == "eos":
        with pytest.raises(StopIteration):
            next(stream)
    elif stop == "close":
        stream.close()
    else:
        monkeypatch.setattr(
            layer, "forward", Mock(side_effect=RuntimeError("forward failed"))
        )
        with pytest.raises(RuntimeError, match="forward failed"):
            next(stream)
    assert layer.layer_past is None
