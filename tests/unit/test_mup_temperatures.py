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

import pytest

from megatron.model.transformer import _get_attention_scaling
from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG


def cpu_arg_config():
    config = deepcopy(BASE_CONFIG)
    config["global_num_gpus"] = 1
    return config


@pytest.mark.cpu
@pytest.mark.parametrize("apply_query_key_layer_scaling", [False, True])
@pytest.mark.parametrize("mup_attn_temp", [1.0, 2.0])
def test_mup_attention_scaling(
    apply_query_key_layer_scaling,
    mup_attn_temp,
):
    head_dim = 64
    norm_factor, coeff = _get_attention_scaling(
        hidden_size_per_attention_head=head_dim,
        layer_number=4,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        use_mup=True,
        mup_attn_temp=mup_attn_temp,
    )

    effective_scale = (coeff if coeff is not None else 1.0) / norm_factor

    assert effective_scale == pytest.approx(1.0 / (head_dim * mup_attn_temp))


@pytest.mark.cpu
@pytest.mark.parametrize("apply_query_key_layer_scaling", [False, True])
def test_standard_attention_scaling_is_unchanged(
    apply_query_key_layer_scaling,
):
    head_dim = 64
    norm_factor, coeff = _get_attention_scaling(
        hidden_size_per_attention_head=head_dim,
        layer_number=4,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        use_mup=False,
        mup_attn_temp=1.0,
    )

    effective_scale = (coeff if coeff is not None else 1.0) / norm_factor

    assert effective_scale == pytest.approx(1.0 / math.sqrt(head_dim))


@pytest.mark.cpu
@pytest.mark.parametrize("name", ["mup_attn_temp", "mup_output_temp"])
def test_mup_temperatures_must_be_positive(name):
    config = cpu_arg_config()
    config["use_mup"] = True
    config[name] = 0.0

    with pytest.raises(
        ValueError,
        match=f"{name} must be greater than zero",
    ):
        NeoXArgs.from_dict(config)


@pytest.mark.cpu
def test_mup_rejects_flash_attention():
    config = cpu_arg_config()
    config["use_mup"] = True
    config["attention_config"] = [[["flash"], "all"]]

    with pytest.raises(
        ValueError,
        match="only supported with global attention",
    ):
        NeoXArgs.from_dict(config)


@pytest.mark.cpu
def test_mup_rejects_transformer_engine_attention():
    config = cpu_arg_config()
    config["use_mup"] = True
    config["te_mha"] = True

    with pytest.raises(
        ValueError,
        match="Transformer Engine attention",
    ):
        NeoXArgs.from_dict(config)
