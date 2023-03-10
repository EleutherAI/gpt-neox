# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
import pathlib
import subprocess

from torch.utils import cpp_extension
from pathlib import Path

srcpath = Path(__file__).parent.absolute()

# Setting this param to a list has a problem of generating different
# compilation commands (with different order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicitly in
# extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load_fused_kernels():
    try:
        import scaled_upper_triang_masked_softmax_cuda
        import scaled_masked_softmax_cuda
    except (ImportError, ModuleNotFoundError) as e:
        print("\n")
        print(e)
        print("=" * 100)
        print(
            f'ERROR: Fused kernels configured but not properly installed. Please run `pip install {str(srcpath)}` to install them'
        )
        print("=" * 100)
        exit()
    return
