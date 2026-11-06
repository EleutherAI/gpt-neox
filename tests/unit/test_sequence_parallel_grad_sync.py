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

from unittest.mock import Mock

import pytest
import torch

from megatron.model import utils as model_utils


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fp32_allreduce", [False, True])
def test_sequence_parallel_grad_hook_dtype(monkeypatch, dtype, fp32_allreduce):
    group = object()
    monkeypatch.setattr(model_utils.mpu, "get_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model_utils.mpu, "get_model_parallel_group", lambda: group)
    monkeypatch.setattr(model_utils.mpu, "get_fp32_allreduce", lambda: fp32_allreduce)

    def fake_all_reduce(gradient, *, group):
        assert gradient.dtype == (torch.float32 if fp32_allreduce else dtype)
        # Simulate summing equal local gradients from two tensor-parallel ranks.
        gradient.mul_(2)

    all_reduce = Mock(side_effect=fake_all_reduce)
    monkeypatch.setattr(model_utils.dist, "all_reduce", all_reduce)

    weight = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    local_gradient = torch.tensor([0.5, -1.0, 2.0], dtype=dtype)
    expected_gradient = local_gradient * 2
    weight.register_hook(model_utils.reduce_weight_grads_from_model_parallel_region)

    (weight * local_gradient).sum().backward()

    all_reduce.assert_called_once()
    assert all_reduce.call_args.kwargs["group"] is group
    assert weight.grad.dtype == dtype
    torch.testing.assert_close(weight.grad, expected_gradient, rtol=0, atol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sequence_parallel_grad_hook_single_rank(monkeypatch, dtype):
    monkeypatch.setattr(model_utils.mpu, "get_model_parallel_world_size", lambda: 1)
    get_fp32_allreduce = Mock(return_value=True)
    monkeypatch.setattr(model_utils.mpu, "get_fp32_allreduce", get_fp32_allreduce)
    all_reduce = Mock()
    monkeypatch.setattr(model_utils.dist, "all_reduce", all_reduce)
    gradient = torch.tensor([0.5, -1.0, 2.0], dtype=dtype)

    result = model_utils.reduce_weight_grads_from_model_parallel_region(gradient)

    assert result is gradient
    get_fp32_allreduce.assert_not_called()
    all_reduce.assert_not_called()
