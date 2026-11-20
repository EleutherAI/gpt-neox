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

pytestmark = pytest.mark.cpu


@pytest.fixture
def adapter():
    # Exercise generate_until without a checkpoint or distributed engine.
    from eval_tasks.eval_adapter import EvalHarnessAdapter

    adapter = EvalHarnessAdapter.__new__(EvalHarnessAdapter)
    adapter._model = Mock()
    adapter.neox_args = SimpleNamespace(recompute=False)
    adapter._max_gen_toks = 4
    adapter.tokenizer = Mock()
    adapter.tokenizer.encode.return_value = [1]
    adapter.cache_hook = Mock()
    adapter.generate = Mock(return_value=[{"text": "answer!"}])
    return adapter


def test_eval_respects_each_requests_sampling_mode(adapter):
    options = [
        {"top_p": 1.0},
        {"do_sample": False, "temperature": 0.7, "min_p": 0.01},
        {"do_sample": True, "min_p": 0.01},
        {"do_sample": True, "temperature": 0.4, "top_p": 0.9},
        {"temperature": 0.6, "until": "!"},
    ]
    requests = [SimpleNamespace(args=("same prompt", opts)) for opts in options]
    assert adapter.generate_until(requests) == ["answer!"] * 4 + ["answer"]
    calls = [call.kwargs for call in adapter.generate.call_args_list]
    assert [kw["temperature"] for kw in calls] == [0.0, 0.0, 1.0, 0.4, 0.0]
    assert calls[2]["min_p"] == 0.01 and calls[3]["top_p"] == 0.9
    assert calls[-1]["stop_tokens"] == [[1]]
    assert [call.args[1] for call in adapter.cache_hook.add_partial.call_args_list] == [
        requests[i].args for i in (0, 1, 4)
    ]
    adapter.model.module.train_mode.assert_called_once()


def test_eval_rejects_zero_temperature_sampling_before_generation(adapter):
    request = SimpleNamespace(args=("prompt", {"do_sample": True, "temperature": 0.0}))
    with pytest.raises(
        ValueError, match="do_sample=True requires a positive temperature"
    ):
        adapter.generate_until([request])
    adapter.generate.assert_not_called()
    adapter.model.module.train_mode.assert_called_once()
