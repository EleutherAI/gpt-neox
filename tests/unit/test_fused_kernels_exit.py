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

import sys

import pytest

from megatron.fused_kernels import load_fused_kernels


@pytest.mark.cpu
def test_missing_fused_kernels_exit_with_error(monkeypatch):
    # A None entry makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "scaled_upper_triang_masked_softmax_cuda", None)
    with pytest.raises(SystemExit) as exit_info:
        load_fused_kernels()
    assert exit_info.value.code == 1
