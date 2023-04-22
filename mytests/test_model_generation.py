# Copyright (c) 2021, EleutherAI
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

"""
instantiate models, save checkpoints, load checkpoints, compare loaded parameters to saved parameters and compare forward pass outputs

This tests contain a relatively large number of functions. They are not split into separate tests because a lot of boilerplate (e.g. instantiate model) needs
to run in order to perform follow up tests. Joining in one test reduces runtime at the expense of decreased transparency of test results in case of failures.
"""


import os
import sys
sys.path.append('/home/lfsm/code/gpt-neox/')
from tests.common import distributed_test, model_setup, parametrize

ymls = ['/home/lfsm/code/gpt-neox/configs/800M.yml', '/home/lfsm/code/gpt-neox/configs/local_setup.yml']
model, optimizer, lr_scheduler, args_loaded = model_setup(ymls)

print(model)