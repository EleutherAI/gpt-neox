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

from copy import deepcopy

import pytest

from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG


def head_dim_config(**overrides):
    config = deepcopy(BASE_CONFIG)
    config.update(overrides)
    config["global_num_gpus"] = max(1, config["model_parallel_size"])
    return config


@pytest.mark.cpu
def test_head_dim_defaults_to_none():
    neox_args = NeoXArgs.from_dict(head_dim_config())

    assert neox_args.head_dim is None


@pytest.mark.cpu
def test_explicit_head_dim_allows_non_divisible_hidden_size():
    neox_args = NeoXArgs.from_dict(
        head_dim_config(
            hidden_size=10,
            num_attention_heads=4,
            head_dim=6,
            pos_emb="none",
        )
    )

    assert neox_args.head_dim == 6


@pytest.mark.cpu
def test_qwen_style_head_dim_is_tensor_parallel_compatible():
    neox_args = NeoXArgs.from_dict(
        head_dim_config(
            hidden_size=1024,
            num_attention_heads=16,
            num_kv_heads=8,
            head_dim=128,
            model_parallel_size=2,
        )
    )

    assert neox_args.head_dim == 128


@pytest.mark.cpu
def test_derived_head_dim_requires_divisible_hidden_size():
    with pytest.raises(
        ValueError,
        match="hidden_size must be divisible by num_attention_heads when head_dim is not set",
    ):
        NeoXArgs.from_dict(
            head_dim_config(
                hidden_size=10,
                num_attention_heads=4,
                pos_emb="none",
            )
        )


@pytest.mark.cpu
@pytest.mark.parametrize("head_dim", [0, -1])
def test_explicit_head_dim_must_be_positive(head_dim):
    with pytest.raises(ValueError, match="head_dim must be greater than 0"):
        NeoXArgs.from_dict(head_dim_config(head_dim=head_dim, pos_emb="none"))


@pytest.mark.cpu
def test_query_heads_must_be_tensor_parallel_compatible():
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by model_parallel_size",
    ):
        NeoXArgs.from_dict(
            head_dim_config(
                hidden_size=12,
                num_attention_heads=3,
                model_parallel_size=2,
                pos_emb="none",
            )
        )


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("head_dim", "rotary_pct", "expected_rotary_ndims"),
    [(5, 1.0, 5), (8, 0.1, 0)],
)
def test_rotary_dimensions_must_be_positive_and_even(
    head_dim, rotary_pct, expected_rotary_ndims
):
    with pytest.raises(
        ValueError,
        match=f"rotary attention dimensions must be a positive even number, got {expected_rotary_ndims}",
    ):
        NeoXArgs.from_dict(
            head_dim_config(head_dim=head_dim, rotary_pct=rotary_pct)
        )


@pytest.mark.cpu
@pytest.mark.parametrize("rotary_pct", [0.0, 1.1])
def test_rotary_pct_must_be_in_supported_range(rotary_pct):
    with pytest.raises(
        ValueError,
        match="rotary_pct must be greater than 0 and less than or equal to 1",
    ):
        NeoXArgs.from_dict(head_dim_config(head_dim=8, rotary_pct=rotary_pct))


@pytest.mark.cpu
def test_mamba_retains_attention_dimension_exemption():
    neox_args = NeoXArgs.from_dict(
        head_dim_config(
            hidden_size=10,
            num_attention_heads=4,
            attention_config=[[["mamba"], "all"]],
        )
    )

    assert neox_args.head_dim is None
