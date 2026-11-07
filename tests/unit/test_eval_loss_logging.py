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

import math
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import call, patch

import pytest

from megatron.neox_arguments import NeoXArgs
from megatron.data.data_utils import get_eval_subset_names, is_separate_eval_enabled
from megatron.training import (
    evaluate_named_data_iterators,
    synthesize_weighted_mix_eval_results,
)
from tests.common import BASE_CONFIG


def cpu_arg_config():
    config = deepcopy(BASE_CONFIG)
    # Avoid hardware discovery in NeoXArgs.calculate_derived during CPU unit tests.
    # Without this, configs with hostfile/include fall through to torch.cuda.device_count()
    # and crash on systems with no visible GPUs.
    config["global_num_gpus"] = 1
    return config


def explicit_eval_path_config(**overrides):
    config = cpu_arg_config()
    config.pop("data_path", None)
    config.update(
        {
            "train_data_paths": ["train_code", "train_math"],
            "valid_data_paths": ["valid_code", "valid_math"],
            "test_data_paths": ["test_code", "test_math"],
            "train_data_weights": [0.7, 0.3],
            "valid_data_weights": [0.7, 0.3],
            "test_data_weights": [0.7, 0.3],
        }
    )
    config.update(overrides)
    return config


@pytest.mark.cpu
def test_eval_loss_logging_defaults_to_blended():
    neox_args = NeoXArgs.from_dict(cpu_arg_config())

    assert neox_args.eval_loss_logging == "blended"
    assert not is_separate_eval_enabled(neox_args)


@pytest.mark.cpu
def test_eval_loss_logging_uses_configured_subset_names():
    neox_args = NeoXArgs.from_dict(
        explicit_eval_path_config(
            eval_loss_logging="separate",
            valid_data_names=["code", "math"],
            test_data_names=["code_eval", "math_eval"],
        )
    )

    assert is_separate_eval_enabled(neox_args)
    assert get_eval_subset_names(neox_args, "valid", 2) == ["code", "math"]
    assert get_eval_subset_names(neox_args, "test", 2) == [
        "code_eval",
        "math_eval",
    ]


@pytest.mark.cpu
def test_eval_loss_logging_invalid_name_length_fails():
    with pytest.raises(ValueError, match="valid_data_names length"):
        NeoXArgs.from_dict(
            explicit_eval_path_config(
                eval_loss_logging="separate",
                valid_data_names=["code"],
            )
        )


@pytest.mark.cpu
def test_eval_loss_logging_rejects_data_path_split_for_separate_mode():
    config = cpu_arg_config()
    config["eval_loss_logging"] = "separate"

    with pytest.raises(ValueError, match="data_path plus split only supports blended"):
        NeoXArgs.from_dict(config)


@pytest.mark.cpu
def test_weighted_mix_aggregates_loss_before_perplexity():
    eval_results = {
        "code": {"lm_loss": 2.0, "lm_loss_ppl": math.exp(2.0)},
        "math": {"lm_loss": 4.0, "lm_loss_ppl": math.exp(4.0)},
    }

    blended = synthesize_weighted_mix_eval_results(
        eval_results, weights={"code": 0.75, "math": 0.25}
    )

    expected_loss = 2.5
    direct_ppl_average = 0.75 * math.exp(2.0) + 0.25 * math.exp(4.0)
    assert blended["lm_loss"] == pytest.approx(expected_loss)
    assert blended["lm_loss_ppl"] == pytest.approx(math.exp(expected_loss))
    assert blended["lm_loss_ppl"] != pytest.approx(direct_ppl_average)


@pytest.mark.cpu
def test_evaluate_named_data_iterators_logs_subsets_and_blended_metric():
    neox_args = SimpleNamespace(
        eval_loss_logging="blended_and_separate",
    )

    def fake_evaluate(*, data_iterator, **_kwargs):
        losses = {"code_iter": 2.0, "math_iter": 4.0}
        loss = losses[data_iterator]
        return {"lm_loss": loss, "lm_loss_ppl": math.exp(loss)}

    with patch("megatron.training.evaluate", side_effect=fake_evaluate), patch(
        "megatron.training.log_eval_results"
    ) as log_eval_results:
        results = evaluate_named_data_iterators(
            neox_args=neox_args,
            prefix="iteration 10",
            forward_step_func=object(),
            named_data_iterators={"code": "code_iter", "math": "math_iter"},
            model=object(),
            iteration=10,
            chart_name="validation",
            weights={"code": 0.75, "math": 0.25},
        )

    assert results["blended"]["lm_loss"] == pytest.approx(2.5)
    assert results["blended"]["lm_loss_ppl"] == pytest.approx(math.exp(2.5))
    log_eval_results.assert_has_calls(
        [
            call(
                neox_args=neox_args,
                prefix="iteration 10",
                total_loss_dict=results["code"],
                iteration=10,
                chart_name="validation/code",
            ),
            call(
                neox_args=neox_args,
                prefix="iteration 10",
                total_loss_dict=results["math"],
                iteration=10,
                chart_name="validation/math",
            ),
            call(
                neox_args=neox_args,
                prefix="iteration 10",
                total_loss_dict=results["blended"],
                iteration=10,
                chart_name="validation/blended",
            ),
        ]
    )
