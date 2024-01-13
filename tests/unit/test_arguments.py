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
from tests.common import BASE_CONFIG, distributed_test


def test_main_constructor():
    input_args = ["train.py", "tests/config/test_setup.yml"]
    neox_args = NeoXArgs.consume_deepy_args(input_args)
    deepspeed_main_args = neox_args.get_deepspeed_main_args()
    neox_args = NeoXArgs.consume_neox_args(input_args=deepspeed_main_args)
    neox_args.configure_distributed_args()


def test_constructor_from_ymls():
    @distributed_test(world_size=[1, 2])
    def _test_constructor_from_ymls():
        neox_args = NeoXArgs.from_ymls(["tests/config/test_setup.yml"])
        neox_args.configure_distributed_args()

    _test_constructor_from_ymls()


def test_constructor_from_dict():
    @distributed_test(world_size=[1, 2])
    def _test_constructor_from_dict():
        neox_args = NeoXArgs.from_dict(BASE_CONFIG)

    _test_constructor_from_dict()
