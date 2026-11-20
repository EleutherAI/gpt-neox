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

from contextlib import contextmanager
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from megatron import mpu
from megatron.model.transformer import ParallelSelfAttention
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG

pytestmark = pytest.mark.cpu


def _make_neox_args(**updates):
    config = deepcopy(BASE_CONFIG)
    config["global_num_gpus"] = 1
    config["use_cpu_initialization"] = True
    config["precision"] = "fp32"
    config.update(updates)
    return NeoXArgs.from_dict(config)


@contextmanager
def _single_rank_model_parallel(tmp_path):
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=0,
        world_size=1,
        timeout=timedelta(seconds=60),
    )
    try:
        mpu.initialize_model_parallel(1)
        yield
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


@pytest.mark.parametrize("use_bias_in_attn_linear", [True, False])
def test_use_bias_in_qkv_defaults_to_use_bias_in_attn_linear(use_bias_in_attn_linear):
    neox_args = _make_neox_args(use_bias_in_attn_linear=use_bias_in_attn_linear)

    assert neox_args.use_bias_in_qkv is use_bias_in_attn_linear


@pytest.mark.parametrize(
    ("use_bias_in_attn_linear", "use_bias_in_qkv"), [(False, True), (True, False)]
)
def test_use_bias_in_qkv_overrides_use_bias_in_attn_linear(
    use_bias_in_attn_linear, use_bias_in_qkv
):
    neox_args = _make_neox_args(
        use_bias_in_attn_linear=use_bias_in_attn_linear,
        use_bias_in_qkv=use_bias_in_qkv,
    )

    assert neox_args.use_bias_in_attn_linear is use_bias_in_attn_linear
    assert neox_args.use_bias_in_qkv is use_bias_in_qkv


@pytest.mark.parametrize("num_kv_heads", [None, 2])
@pytest.mark.parametrize(
    ("use_bias_in_attn_linear", "use_bias_in_qkv", "expect_qkv_bias", "expect_o_bias"),
    [
        (True, None, True, True),
        (False, None, False, False),
        (False, True, True, False),
        (True, False, False, True),
    ],
)
def test_attention_projection_biases(
    tmp_path,
    num_kv_heads,
    use_bias_in_attn_linear,
    use_bias_in_qkv,
    expect_qkv_bias,
    expect_o_bias,
):
    updates = {"use_bias_in_attn_linear": use_bias_in_attn_linear}
    if use_bias_in_qkv is not None:
        updates["use_bias_in_qkv"] = use_bias_in_qkv
    if num_kv_heads is not None:
        updates["num_kv_heads"] = num_kv_heads
    neox_args = _make_neox_args(**updates)

    with _single_rank_model_parallel(tmp_path):
        attention = ParallelSelfAttention(
            neox_args=neox_args,
            attention_mask_func=lambda scores, mask: scores,
            init_method=torch.nn.init.zeros_,
            output_layer_init_method=torch.nn.init.zeros_,
            layer_number=0,
        )

    qkv_bias = attention.query_key_value.bias
    if expect_qkv_bias:
        assert qkv_bias is not None
        assert qkv_bias.shape == (attention.query_key_value.weight.shape[0],)
    else:
        assert qkv_bias is None
    assert (attention.dense.bias is not None) is expect_o_bias
