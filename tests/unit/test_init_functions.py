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
import torch

from megatron.model.init_functions import get_init_methods


@pytest.mark.cpu
@pytest.mark.parametrize(
    "name, n_layers", [("orthogonal", 1), ("scaled_orthogonal", 4)]
)
def test_orthogonal_init_methods(name, n_layers):
    args = SimpleNamespace(
        init_method=name,
        output_layer_init_method=name,
        use_mup=False,
        mup_init_scale=1.0,
        num_layers=4,
    )
    init_method, _ = get_init_methods(args)
    weight = init_method(torch.empty(16, 16))
    torch.testing.assert_close(weight @ weight.T, 2 / n_layers * torch.eye(16))
