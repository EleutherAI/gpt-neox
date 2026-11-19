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
import types

import pytest

from megatron import fused_kernels

KERNELS = [
    "scaled_upper_triang_masked_softmax_cuda",
    "scaled_masked_softmax_cuda",
    "fused_rotary_positional_embedding",
]


@pytest.mark.cpu
def test_load_makes_jit_kernels_importable(monkeypatch, tmp_path):
    for name in KERNELS:
        # Lets monkeypatch remove the modules that load() registers.
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(
        fused_kernels.cpp_extension,
        "load",
        lambda name, **kwargs: types.ModuleType(name),
    )
    monkeypatch.setattr(
        fused_kernels, "_get_cuda_bare_metal_version", lambda _: ("", "12", "4")
    )

    fused_kernels.load(types.SimpleNamespace(fused_kernels_build_path=tmp_path, rank=0))
    fused_kernels.load_fused_kernels()
