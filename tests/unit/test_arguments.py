# Copyright (c) 2025, EleutherAI
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

import logging
from copy import deepcopy

import pytest

from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG, DistributedTest


def _cpu_moe_config(**overrides):
    config = deepcopy(BASE_CONFIG)
    # Avoid hardware discovery in NeoXArgs.calculate_derived during CPU unit tests.
    # Without this, configs with hostfile/include fall through to torch.cuda.device_count()
    # and crash on systems with no visible GPUs.
    config["global_num_gpus"] = 1
    config.update(overrides)
    return config


@pytest.mark.cpu
def test_moe_topk_router_warns_no_load_balancing(caplog):
    """
    TopKTokenChoiceRouter has no load balancing loss, so configuring
    moe_router_type: "topk" for a training run silently produces a
    misconfigured MoE (see issue #1364). Validation must not stay silent.
    """
    with caplog.at_level(logging.WARNING):
        NeoXArgs.from_dict(_cpu_moe_config(moe_num_experts=2, moe_router_type="topk"))
    assert any(
        "load balancing" in record.getMessage() for record in caplog.records
    ), "expected a warning that the top-k router does not apply a load balancing loss"


@pytest.mark.cpu
def test_moe_sinkhorn_router_does_not_warn(caplog):
    """The supported training router (sinkhorn) must not trigger the warning."""
    with caplog.at_level(logging.WARNING):
        NeoXArgs.from_dict(
            _cpu_moe_config(moe_num_experts=2, moe_router_type="sinkhorn")
        )
    assert not any("load balancing" in record.getMessage() for record in caplog.records)


def test_main_constructor():
    input_args = [
        "train.py",
        "tests/config/test_setup.yml",
        "configs/cpu_mock_config.yml",
    ]
    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    neox_args = NeoXArgs.consume_neox_args(input_args=deepspeed_main_args)
    neox_args.configure_distributed_args()


class test_constructor_from_ymls_class(DistributedTest):
    world_size = 2

    def test(self):
        neox_args = NeoXArgs.from_ymls(
            ["tests/config/test_setup.yml", "configs/cpu_mock_config.yml"]
        )
        neox_args.configure_distributed_args()


def test_constructor_from_ymls():
    t1 = test_constructor_from_ymls_class()
    t1.test()


class test_constructor_from_dict_class(DistributedTest):
    world_size = 2

    def test(self):
        config = BASE_CONFIG.copy()
        config["global_num_gpus"] = 1
        neox_args = NeoXArgs.from_dict(config)


def test_constructor_from_dict():
    t1 = test_constructor_from_dict_class()
    t1.test()
