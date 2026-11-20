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

pytestmark = pytest.mark.cpu


def test_standalone_filter_validates_before_modifying_logits():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="epsilon_cutoff"):
        generation.filter_logits(logits, top_k=1, epsilon_cutoff=1.0)
    torch.testing.assert_close(logits, torch.tensor([[1.0, 2.0]]))
    result = generation.filter_logits(logits, 1, 0.0, -100.0)
    torch.testing.assert_close(result, torch.tensor([[-100.0, 2.0]]))


def test_gather_generation_logits(monkeypatch):
    # Recompute can select a non-contiguous [batch, shard] view.
    shard = torch.arange(24).reshape(2, 3, 4)[:, 1, :]
    assert not shard.is_contiguous()
    full = torch.cat((shard, shard + 100), dim=-1)
    gather = Mock(return_value=full)
    monkeypatch.setattr(generation, "gather_from_model_parallel_region", gather)
    assert generation._gather_generation_logits(shard, 8, 2) is full
    (sent,) = gather.call_args.args
    torch.testing.assert_close(sent, shard)
    assert sent.is_contiguous()

    # A head that already gathers must not double the vocabulary or communicate.
    assert generation._gather_generation_logits(full, 8, 2) is full
    gather.assert_called_once()


def test_gather_generation_logits_rejects_wrong_width():
    with pytest.raises(ValueError, match="width 3 for TP=2"):
        generation._gather_generation_logits(torch.empty(1, 3), 8, 2)


@pytest.fixture
def args(monkeypatch):
    # A single rank needs no communication.
    monkeypatch.setattr(generation.mpu, "get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(generation.mpu, "get_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(generation.mpu, "get_model_parallel_src_rank", lambda: 0)
    monkeypatch.setattr(generation.mpu, "get_model_parallel_group", lambda: None)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    return SimpleNamespace(
        tokenizer=SimpleNamespace(eod=0),
        seq_length=16,
        padded_vocab_size=2,
        eod_mask_loss=False,
        sliding_window_width=None,
        is_pipe_parallel=False,
        return_logits=True,
    )


def test_pipeline_forward_preserves_inputs_and_restores_engine_on_error():
    engine = Mock()
    original_batch_fn = object()
    engine.batch_fn = original_batch_fn
    engine.total_loss = 17
    engine.monitor.enabled = True
    inputs = (torch.tensor([[7]]), torch.tensor([[4]]), torch.zeros(1, 1, 1, 5))

    def fail(batches, **kwargs):
        batch_inputs, labels = next(batches)
        assert batch_inputs is inputs and labels == ()
        assert kwargs["compute_loss"] is False and kwargs["num_micro_batches"] == 1
        assert engine.total_loss is None and not engine.monitor.enabled
        raise RuntimeError("pipeline forward failed")

    engine.eval_batch.side_effect = fail
    with pytest.raises(RuntimeError, match="pipeline forward failed"):
        generation.forward_model(engine, inputs, is_pipe_parallel=True)
    assert engine.set_batch_fn.call_args_list == [((None,),), ((original_batch_fn,),)]
    assert engine.total_loss == 17 and engine.monitor.enabled


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
            Mock(device=torch.device("cpu"), module=Mock(side_effect=position_logits)),
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
    return Mock(device=torch.device("cpu"), module=module), layer


def test_prompt_generation_honors_stop_sequence_on_cpu(args, cached_model):
    args.tokenizer.tokenize = lambda _: [0]
    args.tokenizer.detokenize = lambda tokens: "".join(map(str, tokens))
    model, _ = cached_model  # Generates token 1 at every step.
    results = generation.generate_samples_from_prompt(
        args, model, ["first", "second"], maximum_tokens=4, stop_tokens=[[1, 1]]
    )
    assert [result["context"] for result in results] == ["first", "second"]
    assert all(result["finished"] and len(result["logits"]) == 2 for result in results)


def test_sampling_validation_is_outside_the_decode_loop(
    args, cached_model, monkeypatch
):
    validate = Mock(wraps=generation.validate_sampling_options)
    monkeypatch.setattr(generation, "validate_sampling_options", validate)
    model, _ = cached_model
    steps = list(
        generation.stream_tokens(args, model, [[1]], maximum_tokens=3, top_k=1)
    )
    assert len(steps) == 3
    validate.assert_called_once()


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
