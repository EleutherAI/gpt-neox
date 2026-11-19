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

import pytest
import torch

from megatron.neox_arguments import NeoXArgs
from megatron.training import get_optimizer
from tests.common import get_configs_with_path


@pytest.mark.cpu
def test_cpu_torch_adam_uses_torch_adam():
    neox_args = NeoXArgs.from_ymls(
        get_configs_with_path(["125M.yml", "local_setup.yml", "cpu_mock_config.yml"]),
        overwrite_values={
            "optimizer": {"type": "cpu_torch_adam", "params": {"lr": 1e-3}}
        },
    )
    optimizer, _ = get_optimizer(torch.nn.Linear(4, 4), neox_args)
    assert type(optimizer) is torch.optim.Adam
