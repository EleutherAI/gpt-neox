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
import torch.nn.functional as F

from megatron import mpu
from megatron.model.transformer import ParallelMLP
from megatron.neox_arguments import NeoXArgs


@pytest.mark.cpu
def test_bias_free_unfused_gelu_forward_and_backward(monkeypatch):
    monkeypatch.setattr(mpu.initialize, "_MPU_WORLD_SIZE", 1)
    monkeypatch.setattr(mpu.initialize, "_MPU_RANK", 0)
    args = NeoXArgs(
        global_num_gpus=1,
        train_micro_batch_size_per_gpu=1,
        use_cpu_initialization=True,
        num_layers=1,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        max_position_embeddings=4,
        activation="gelu",
        bias_gelu_fusion=False,
        use_bias_in_mlp=False,
    )
    mlp = ParallelMLP(
        args,
        init_method=torch.nn.init.xavier_uniform_,
        output_layer_init_method=torch.nn.init.xavier_uniform_,
        multiple_of=args.mlp_multiple_of,
    )
    x = torch.randn(4, 2, args.hidden_size, requires_grad=True)

    output, output_bias = mlp(x)
    assert output_bias is None
    projected, projection_bias = mlp.linear1(x)
    assert projection_bias is None
    expected, _ = mlp.linear2(F.gelu(projected))
    torch.testing.assert_close(output, expected)

    inputs = (x, mlp.linear1.weight, mlp.linear2.weight)
    grad_output = torch.randn_like(output)
    torch.testing.assert_close(
        torch.autograd.grad(output, inputs, grad_output),
        torch.autograd.grad(expected, inputs, grad_output),
    )
