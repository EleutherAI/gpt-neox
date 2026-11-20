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

import json
import warnings
from datetime import timedelta
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron import text_generation_utils as generation
from megatron.model.utils import SequentialWrapper
from megatron.neox_arguments import sampling

pytestmark = pytest.mark.cpu


def test_standalone_filter_validates_before_modifying_logits():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError, match="epsilon_cutoff"):
        generation.filter_logits(logits, top_k=1, epsilon_cutoff=1.0)
    torch.testing.assert_close(logits, torch.tensor([[1.0, 2.0]]))
    result = generation.filter_logits(logits, 1, 0.0, -100.0)
    torch.testing.assert_close(result, torch.tensor([[-100.0, 2.0]]))


def test_oversize_top_k_uses_full_vocab_and_still_applies_top_p():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.warns(UserWarning, match=r"top_k=99.*truncating top_k to 2"):
        result = generation.filter_logits(logits, top_k=99, top_p=0.5)
    torch.testing.assert_close(result, torch.tensor([[-float("inf"), 2.0]]))


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


def test_overlong_row_is_only_truncated_when_requested():
    rows = [[1], [2, 3, 4]]
    with pytest.raises(ValueError, match="3 tokens.*2 tokens"):
        generation.pad_batch(rows, pad_id=0, pad_len=2)
    assert rows == [[1], [2, 3, 4]]
    assert generation.pad_batch(rows, pad_id=0, pad_len=2, truncate=True) == (
        [[1, 0], [2, 3]], [1, 2]
    )


def _check_tp_prompt_recovery(rank, rendezvous):
    dist = torch.distributed
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        generation.mpu.initialize_model_parallel(2)
        owner = generation.mpu.get_model_parallel_rank() == 0
        args = SimpleNamespace(
            seq_length=4,
            padded_vocab_size=2,
            is_pipe_parallel=False,
            return_logits=False,
            eod_mask_loss=False,
            sliding_window_width=None,
            tokenizer=SimpleNamespace(
                eod=0,
                tokenize=lambda s: [1] * len(s),
                detokenize=lambda ids: "".join(map(str, ids)),
            ),
        )

        def logits(inputs):
            tokens = inputs[0].float()
            return torch.stack((tokens, tokens + 1), dim=-1)  # Always generate ID 1.

        model = Mock(device=torch.device("cpu"), module=Mock(side_effect=logits))
        generate = partial(
            generation.generate_samples_from_prompt, args, model, maximum_tokens=1
        )
        # TP group [0, 1] rejects its owner's request. Group [2, 3] must not
        # share that error: its own prompt exactly fills the context window.
        if rank < 2:
            with pytest.raises(ValueError) as exc:
                generate("12345" if owner else [])
            assert str(exc.value) == (
                "Prompt context_length (5 tokens) exceeds seq_length (4 tokens)."
            )
        else:
            records = generate("1234" if owner else [])
            assert [r["length"] for r in records] == ([0] if owner else [])
        model.module.assert_not_called()

        # A bad later row in the raw API is rejected on every rank, too.
        with pytest.raises(ValueError) as exc:
            list(generation.stream_tokens(args, model, [[1], [1] * 6] if owner else []))
        assert str(exc.value) == (
            "Prompt context_length (6 tokens) exceeds seq_length (4 tokens)."
        )
        model.module.assert_not_called()

        # Reuse the same groups and model. Non-owners' placeholders are not
        # validated, and both the ragged row lengths and prompt IDs come from the owner.
        prompts = [[0], [1, 0, 1]] if rank < 2 else [[1, 0], [0]]
        steps = list(
            generation.stream_tokens(
                args, model, prompts if owner else [[0] * 9], maximum_tokens=1
            )
        )
        tokens, start, *_ = steps[-1]
        assert start.tolist() == list(map(len, prompts))
        assert [
            row[: len(prompt)].tolist() for row, prompt in zip(tokens, prompts)
        ] == prompts
        records = generate("ok" if owner else [])
        assert [r["text"] for r in records] == (["1"] if owner else [])
    finally:
        generation.mpu.destroy_model_parallel()
        dist.destroy_process_group()


def test_overlong_prompt_is_recoverable_and_scoped_to_each_tp_group(tmp_path):
    torch.multiprocessing.spawn(
        _check_tp_prompt_recovery,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=4,
    )


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


@pytest.mark.parametrize("exit_kind", ["exhaust", "eos", "close", "error", "no_steps"])
def test_pipeline_stream_releases_activation_shapes(
    args, cached_model, monkeypatch, exit_kind
):
    args.is_pipe_parallel = True
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *args, **kwargs: None
    )
    model, _ = cached_model
    model.num_stages = 1
    model.grid.stage_to_global.return_value = 0
    model.pipe_recv_buf = (1, args.seq_length)  # Shape cached by a normal batch.
    model.reset_activation_shape.side_effect = lambda: setattr(
        model, "pipe_recv_buf", None
    )

    def eval_batch(batches, **kwargs):
        inputs, _ = next(batches)
        assert model.pipe_recv_buf is None  # Cannot reuse the previous batch's shape.
        model.pipe_recv_buf = tuple(inputs[0].shape)
        if exit_kind == "error" and model.pipe_recv_buf == (1, 1):
            raise RuntimeError("decode forward failed")
        return None, model.module(inputs)

    model.eval_batch.side_effect = eval_batch
    stream = generation.stream_tokens(
        args,
        model,
        [[1, 1]],
        maximum_tokens=0 if exit_kind == "no_steps" else 3,
        eos_token_id=1 if exit_kind == "eos" else 0,
        minimum_tokens=1 if exit_kind == "eos" else 0,
    )
    if exit_kind == "no_steps":
        assert list(stream) == []
        assert model.pipe_recv_buf == (1, args.seq_length)
        return
    if exit_kind in ("exhaust", "eos"):
        assert len(list(stream)) == (2 if exit_kind == "eos" else 3)
    elif exit_kind == "close":
        next(stream)  # Prefill
        next(stream)  # One-token decode
        stream.close()
    else:
        next(stream)
        with pytest.raises(RuntimeError, match="decode forward failed"):
            next(stream)
    assert model.pipe_recv_buf is None


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


@pytest.mark.parametrize(
    "filters",
    [
        {"min_p": 0.01},
        {"top_p": 1.0},
        dict(
            top_k=1,
            top_p=0.5,
            min_p=0.9,
            top_h=0.4,
            typical_p=0.2,
            epsilon_cutoff=0.3,
            eta_cutoff=0.3,
        ),
    ],
    ids=["reported-min-p", "neutral-top-p", "all-filters"],
)
def test_zero_temperature_ignores_filters_and_rng(args, cached_model, filters):
    model, _ = cached_model
    with torch.random.fork_rng(devices=[]):
        for seed in range(20):
            torch.random.default_generator.manual_seed(seed)
            before = torch.get_rng_state()
            (step,) = generation.stream_tokens(
                args, model, [[1]], temperature=0.0, maximum_tokens=1, **filters
            )
            tokens, start, _, scores, _ = step
            assert tokens[0, start.item()].item() == 1
            torch.testing.assert_close(scores, torch.tensor([[1.0, 2.0]]))
            assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("top_p", [0.0, 1.0])
def test_neutral_top_p_skips_nucleus_filtering(monkeypatch, top_p):
    # Both the legacy and HF sentinels leave all logits alone, without sorting.
    monkeypatch.setattr(torch, "sort", Mock(side_effect=AssertionError("sorted")))
    logits = torch.tensor([[3.0, 1.0, -90.0]])
    torch.testing.assert_close(
        generation.filter_logits(logits.clone(), top_p=top_p), logits
    )


@pytest.mark.parametrize("top_p,count", [(0.49, 2), (0.5, 2), (0.51, 3)])
def test_top_p_keeps_shortest_prefix_reaching_mass(top_p, count):
    # Test two independent rows: uniform probabilities and a unique top token.
    logits = torch.tensor([[0.0] * 4, [10.0, 0.0, -10.0, 0.0]])
    kept = generation.filter_logits(logits, top_p=top_p).isfinite()
    assert kept[0].sum().item() == count
    assert kept[1].tolist() == [True, False, False, False]


@pytest.mark.parametrize("temperature", [1e-37, 1e-300, 5e-324])
def test_cold_sampling_preserves_ties_masks_and_distinct_maxima(temperature):
    logits = torch.tensor(
        [[100.0, 100.0, 90.0], [-101.0, -100.0, -float("inf")], [0.0, 0.0, 0.0]]
    )
    scores = generation._scale_sampling_logits(logits, temperature)
    assert scores.dtype == torch.float32 and scores[1, 2].isneginf()
    torch.testing.assert_close(
        scores.softmax(-1),
        torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [1 / 3] * 3]),
    )


@pytest.mark.parametrize(
    "logits,temperature",
    [
        ([0.0, 1e-39, 2e-39], 1e-39),
        ([-float("inf"), -3.0, 4.0], 1e300),
        ([-3.0, 0.0, 4.0], 0.25),
        ([-3.0, 0.0, 4.0], 1.0),
        ([-3.0, 0.0, 4.0], 2.0),
    ],
)
def test_temperature_preserves_representable_scores(logits, temperature):
    logits = torch.tensor([logits])
    expected = (logits.double() / temperature).float()
    torch.testing.assert_close(
        generation._scale_sampling_logits(logits, temperature), expected
    )


def test_positive_temperature_samples_scaled_filtered_logits(
    args, cached_model, monkeypatch
):
    model, _ = cached_model
    draw = Mock(wraps=torch.multinomial)
    monkeypatch.setattr(torch, "multinomial", draw)
    (step,) = generation.stream_tokens(
        args, model, [[1]], maximum_tokens=1, temperature=0.5, top_p=0.7
    )
    draw.assert_called_once()
    probabilities, _ = draw.call_args.args
    torch.testing.assert_close(probabilities, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(step[3], torch.tensor([[-float("inf"), 4.0]]))


def test_prompt_generation_honors_stop_sequence_on_cpu(args, cached_model):
    args.tokenizer.tokenize = lambda _: [0]
    args.tokenizer.detokenize = lambda tokens: "".join(map(str, tokens))
    model, _ = cached_model  # Generates token 1 at every step.
    results = generation.generate_samples_from_prompt(
        args, model, ["first", "second"], maximum_tokens=4, stop_tokens=[[1, 1]]
    )
    assert [result["context"] for result in results] == ["first", "second"]
    assert all(result["finished"] and len(result["logits"]) == 2 for result in results)
    assert [(result["text"], result["length"]) for result in results] == [("11", 2)] * 2


@pytest.mark.parametrize("recompute", [False, True], ids=["cached", "recompute"])
@pytest.mark.parametrize(
    "prompt,draws,until,expected,steps",
    [
        ("4", "1234", "23", "1", 3),         # Two-token delimiter after an answer.
        ("4", "234", "23", "", 2),           # The completion starts with the delimiter.
        ("4", "1234", ["4", "3"], "12", 3),  # Alternative single-token delimiter.
        ("4", "1234", ["3", "23"], "1", 3),  # Two suffixes match on the same step.
        ("2", "3104", "23", "31", 3),        # The prompt's 2 cannot start a match.
        ("4", "1204", "23", "12", 3),        # EOS after a partial match; keep the 2.
        ("4", "12", "23", "12", 2),          # Token budget after a partial match.
        ("4", "01", "23", "", 1),            # EOS on the first step stays excluded.
    ],
)
def test_lm_eval_strips_complete_stop(
    args, recompute, prompt, draws, until, expected, steps
):
    from eval_tasks.eval_adapter import EvalHarnessAdapter

    args.tokenizer.tokenize = args.tokenizer.encode = lambda text: list(map(int, text))
    args.tokenizer.detokenize = lambda ids: "".join(map(str, ids))
    args.recompute = recompute
    args.padded_vocab_size = 5
    sequence = iter(map(int, draws))

    def next_scores(inputs):
        scores = torch.zeros(*inputs[0].shape, args.padded_vocab_size)
        scores[..., next(sequence)] = 1
        return scores

    model = Mock(device=torch.device("cpu"), module=Mock(side_effect=next_scores))
    adapter = EvalHarnessAdapter.__new__(EvalHarnessAdapter)
    adapter._model, adapter.neox_args, adapter.tokenizer = model, args, args.tokenizer
    adapter._max_gen_toks, adapter.cache_hook = len(draws), Mock()
    adapter.generate = partial(generation.generate_samples_from_prompt, args, model)
    request = SimpleNamespace(args=(prompt, {"until": until}))
    assert adapter.generate_until([request]) == [expected]
    assert model.module.call_count == steps
    adapter.cache_hook.add_partial.assert_called_once_with(
        "generate_until", request.args, expected
    )


@pytest.mark.parametrize("recompute", [False, True], ids=["cached", "recompute"])
def test_stop_suffix_must_be_generated_by_each_row(args, cached_model, recompute):
    model, _ = cached_model  # Generates only ones.
    steps = list(
        generation.stream_tokens(
            args,
            model,
            [[0], [1, 1, 1]],
            maximum_tokens=5,
            recompute=recompute,
            stop_tokens=[[], [1, 1]],
        )
    )
    assert [done.tolist() for *_, done in steps] == [
        [False, False],  # Second row contains a stop inside its prompt.
        [True, False],  # Only the first row has generated two ones.
        [True, False],  # A prompt token plus one generated token is not a stop.
        [True, True],
    ]
    # The first row's inclusive end stays put while the second finishes.
    tokens, starts, ends, *_ = steps[-1]
    assert ends.tolist() == [2, 4]
    assert [row[a : b + 1].tolist() for row, a, b in zip(tokens, starts, ends)] == [
        [1, 1],
        [1, 1],
    ]


@pytest.mark.parametrize(
    "options,message",
    [
        (
            dict(temperature=0.0, top_p=0.9, min_p=0.01),
            r"will be ignored: top_p=0.9, min_p=0.01",
        ),
        (dict(temperature=1.0, top_k=99), r"truncating top_k to 2"),
        (dict(temperature=0.0, top_p=1.0), None),
        (dict(temperature=0.0, top_p=0.0), None),
        (dict(temperature=1.0, top_k=2), None),
    ],
)
def test_multi_prompt_request_validates_and_warns_once(
    args, cached_model, monkeypatch, options, message
):
    args.tokenizer.tokenize = lambda _: [1]
    args.tokenizer.detokenize = lambda tokens: "".join(map(str, tokens))
    model, _ = cached_model
    validate = Mock(wraps=sampling.validate_sampling_options)
    monkeypatch.setattr(sampling, "validate_sampling_options", validate)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = generation.generate_samples_from_prompt(
            args, model, ["one", "two"], minimum_tokens=2, maximum_tokens=3, **options
        )
    assert len(rows) == 2
    validate.assert_called_once()
    messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(messages) == int(message is not None)
    if message:
        assert message in messages[0]


@pytest.mark.parametrize("mode", ["input-file", "unconditional", "interactive"])
def test_cli_uses_config_without_duplicate_checks_or_warnings(
    mode, args, cached_model, monkeypatch, tmp_path, capsys
):
    import generate as cli
    from megatron.neox_arguments import NeoXArgs
    from tests.common import get_configs_with_path

    prompts, output = tmp_path / "prompts", tmp_path / "answers"
    prompts.write_text("one\ntwo\n")
    model, _ = cached_model
    args.tokenizer.tokenize = lambda _: [1]
    args.tokenizer.detokenize = lambda ids: "".join(map(str, ids))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        torch.distributed, "broadcast_object_list", lambda *a, **kw: None
    )
    monkeypatch.setattr(generation.os, "system", lambda _: 0)
    monkeypatch.setattr("builtins.input", Mock(side_effect=["one", EOFError()]))
    validate = Mock(wraps=sampling.validate_sampling_options)
    monkeypatch.setattr(sampling, "validate_sampling_options", validate)
    sample = mode == "unconditional"
    warning = "truncating top_k to 2" if sample else "will be ignored: top_k=1"
    with pytest.warns(UserWarning, match=warning) as caught:
        config = NeoXArgs.from_ymls(
            get_configs_with_path(
                ["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]
            ),
            overwrite_values=dict(
                log_dir=None,
                text_gen_type=mode,
                sample_input_file=str(prompts),
                sample_output_file=str(output),
                num_samples=2,
                pipe_parallel_size=0,
                padded_vocab_size=2,
                seq_length=16,
                minimum_tokens=2,
                maximum_tokens=2,
                temperature=1.0 if sample else 0.0,
                top_k=99 if sample else 1,
            ),
        )
        config.tokenizer = args.tokenizer
        monkeypatch.setattr(
            cli, "setup_for_inference_or_eval", lambda **kw: (model, config)
        )
        cli.main()
    assert sum(warning in str(w.message) for w in caught) == 1
    validate.assert_called_once()
    if mode == "interactive":
        assert "Generated Text: 11" in capsys.readouterr().out
    else:
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert [r["context"] for r in rows] == (
            ["one", "two"] if mode == "input-file" else ["", ""]
        )
        assert [r["text"] for r in rows] == ["11", "11"]


def test_precompute_and_unknown_modes_do_not_build_a_sampling_stream(monkeypatch):
    precompute = Mock()
    monkeypatch.setattr(generation, "precompute_logits", precompute)
    args, model = SimpleNamespace(text_gen_type="precompute"), Mock()
    generation.generate_from_config(args, model)
    precompute.assert_called_once_with(args, model)
    args.text_gen_type = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        generation.generate_from_config(args, model)


def test_early_eos_file_has_only_executed_processed_scores(
    args, cached_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    args.tokenizer.tokenize = lambda _: [1]
    args.tokenizer.detokenize = lambda tokens: "".join(map(str, tokens))
    prompts, output = tmp_path / "in.txt", tmp_path / "out.jsonl"
    prompts.write_text("x\n")
    model, _ = cached_model
    generation.generate_samples_input_from_file(
        args,
        model,
        prompts,
        output_file=output,
        eos_token_id=1,
        minimum_tokens=2,
        maximum_tokens=8,
    )
    row = json.loads(output.read_text())
    assert row["finished"] and row["length"] == 2
    scores = torch.tensor(row["logits"])
    assert scores.shape == (
        3,
        2,
    )  # Two generated tokens and the EOS decision, not eight.
    assert scores[:2, 1].isneginf().all() and scores[2, 1].isfinite()


def test_minimum_cannot_exceed_explicit_or_inferred_maximum(args, cached_model):
    model, _ = cached_model
    with pytest.raises(ValueError, match=r"minimum_tokens \(4\).*maximum_tokens \(3\)"):
        generation.stream_tokens(args, model, [[1]], minimum_tokens=4, maximum_tokens=3)
    model.eval.assert_not_called()
    with pytest.raises(ValueError, match="inferred maximum_tokens"):
        next(
            generation.stream_tokens(
                args, model, [[1] * (args.seq_length - 1)], minimum_tokens=2
            )
        )


def test_sampling_validation_is_outside_the_decode_loop(
    args, cached_model, monkeypatch
):
    validate = Mock(wraps=sampling.validate_sampling_options)
    monkeypatch.setattr(sampling, "validate_sampling_options", validate)
    model, _ = cached_model
    steps = list(
        generation.stream_tokens(
            args, model, [[1]], maximum_tokens=3, temperature=1.0, top_k=1
        )
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
