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

from setuptools import setup, find_packages
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
import subprocess


def _get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

class CommandMixin(object):
    user_options = [
        ('head_size=', 64, 'head size for the kernel'),
        ('max_seq_length=', 512, 'maximum  sequence length for the kernel')
    ]

    def initialize_options(self):
        super().initialize_options()
        # Initialize options
        self.head_size = 64
        self.max_seq_length = 512

    def finalize_options(self):
        # Validate options
        if self.head_size <= 0:
            raise ValueError("head_size must be positive")
        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        super().finalize_options()

    def run(self):
        # Use options
        global head_size, max_seq_length
        head_size = self.head_size
        max_seq_length = self.max_seq_length
        global cuda_ext_args 
        cuda_ext_args = ["-res-usage", 
              "--use_fast_math", 
              "-O3", "-Xptxas -O3", 
              "--extra-device-vectorization", 
              f"-D_N_={head_size}", 
              f"-D_T_={max_seq_length}"]
        print("here")
        super().run()

class ExtensionCommand(CommandMixin, BuildExtension):
    user_options = getattr(BuildExtension, 'user_options', []) + CommandMixin.user_options

srcpath = Path(__file__).parent.absolute()

setup(
    name="rwkv_cuda",
    include_package_data=False,
    ext_modules=[
        CUDAExtension(
            name="wkv6_cuda",
            sources=[
                str(srcpath / "wkv6_op.cpp"),
                str(srcpath / "wkv6_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        )
    ],
    cmdclass={"build_ext": ExtensionCommand},
)
