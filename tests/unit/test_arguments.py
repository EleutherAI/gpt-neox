# Copyright (c) 2024, EleutherAI
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

from megatron.neox_arguments import NeoXArgs
from tests.common import BASE_CONFIG, DistributedTest


def test_main_constructor():
    input_args = ["train.py", "tests/config/test_setup.yml"]
    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    neox_args = NeoXArgs.consume_neox_args(input_args=deepspeed_main_args)
    neox_args.configure_distributed_args()


class test_constructor_from_ymls_class(DistributedTest):
    world_size = 2

    def test(self):
        neox_args = NeoXArgs.from_ymls(["tests/config/test_setup.yml"])
        neox_args.configure_distributed_args()


def test_constructor_from_ymls():
    t1 = test_constructor_from_ymls_class()
    t1.test()


class test_constructor_from_dict_class(DistributedTest):
    world_size = 2

    def test(self):
        neox_args = NeoXArgs.from_dict(BASE_CONFIG)


def test_constructor_from_dict():
    t1 = test_constructor_from_dict_class()
    t1.test()
