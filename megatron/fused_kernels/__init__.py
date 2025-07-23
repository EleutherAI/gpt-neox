# Copyright (c) 2025, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#
# This file has been modified from its original version
#

import os
import pathlib
import subprocess
import torch
import time
import sys
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating different
# compilation commands (with different order of architectures) and
# leading to recompilation of fused kernels. Set it to empty string
# to avoid recompilation and assign arch flags explicitly in
# extra_cuda_cflags below

# TODO: Remove this
# os.environ["TORCH_CUDA_ARCH_LIST"] = ""


def load(neox_args=None):
    print("\n" + "="*80)
    print("FUSED KERNELS: Starting fused kernel loading process...")
    print("="*80)
    start_time = time.time()

    # Check if cuda 11 is installed for compute capability 8.0
    cc_flag = []
    if torch.version.hip is None:
        print(f"FUSED KERNELS: Detected PyTorch with CUDA support")
        print(f"FUSED KERNELS: CUDA_HOME = {cpp_extension.CUDA_HOME}")

        raw_output, bare_metal_major, bare_metal_minor = _get_cuda_bare_metal_version(
            cpp_extension.CUDA_HOME
        )
        print(f"FUSED KERNELS: Detected CUDA version {bare_metal_major}.{bare_metal_minor}")

        if int(bare_metal_major) >= 11:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_80,code=sm_80")
            print(f"FUSED KERNELS: Added compute capability 8.0 (A100)")

            if int(bare_metal_minor) >= 1:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_86,code=sm_86")
                print(f"FUSED KERNELS: Added compute capability 8.6")
            elif int(bare_metal_minor) >= 4:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_87,code=sm_87")
                print(f"FUSED KERNELS: Added compute capability 8.7")
            elif int(bare_metal_minor) >= 8:
                cc_flag.append("-gencode")
                cc_flag.append("arch=compute_89,code=sm_89")
                print(f"FUSED KERNELS: Added compute capability 8.9")
        if int(bare_metal_major) >= 12:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
            print(f"FUSED KERNELS: Added compute capability 9.0 (H100)")
    else:
        print(f"FUSED KERNELS: Detected PyTorch with ROCm/HIP support")

    # Build path
    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / "build"
    print(f"FUSED KERNELS: Source path = {srcpath}")
    print(f"FUSED KERNELS: Build path = {buildpath}")
    _create_build_dir(buildpath)

    # Determine verbosity
    verbose = True if neox_args is None else (neox_args.rank == 0)

    # Helper function to build the kernels.
    def _cpp_extention_load_helper(
        name, sources, extra_cuda_flags, extra_include_paths
    ):
        if torch.version.hip is not None:
            extra_cuda_cflags = ["-O3"] + extra_cuda_flags + cc_flag
        else:
            extra_cuda_cflags = (
                ["-O3", "-gencode", "arch=compute_70,code=sm_70", "--use_fast_math"]
                + extra_cuda_flags
                + cc_flag
            )

        # Check if kernel is already built
        kernel_path = buildpath / name
        if os.path.exists(kernel_path) and any(f.endswith('.so') for f in os.listdir(kernel_path) if os.path.isfile(os.path.join(kernel_path, f))):
            print(f"FUSED KERNELS: {name} appears to be already built in {kernel_path}")
            print(f"FUSED KERNELS: Loading existing compiled kernel...")
        else:
            print(f"FUSED KERNELS: {name} needs to be built")
            print(f"FUSED KERNELS: This will take 30-60 seconds...")
            print(f"FUSED KERNELS: Building with flags: {extra_cuda_cflags}")

        sys.stdout.flush()  # Force flush to ensure messages appear

        try:
            print(f"FUSED KERNELS: Calling cpp_extension.load for {name}...")
            build_start = time.time()

            # Monkey-patch the ninja build to add progress messages
            original_build = cpp_extension._write_ninja_file_and_build_library
            def build_with_progress(*args, **kwargs):
                print(f"FUSED KERNELS: JIT compiling {name} with ninja...")
                print(f"FUSED KERNELS: This involves compiling CUDA kernels - please be patient...")
                sys.stdout.flush()
                return original_build(*args, **kwargs)

            cpp_extension._write_ninja_file_and_build_library = build_with_progress

            try:
                loaded_module = cpp_extension.load(
                    name=name,
                    sources=sources,
                    build_directory=buildpath,
                    extra_cflags=[
                        "-O3",
                    ],
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    verbose=verbose,
                )
            finally:
                # Restore original function
                cpp_extension._write_ninja_file_and_build_library = original_build

            build_time = time.time() - build_start
            print(f"FUSED KERNELS: Successfully loaded {name} in {build_time:.2f} seconds")
            return loaded_module

        except Exception as e:
            print(f"\nFUSED KERNELS ERROR: Failed to build/load {name}")
            print(f"FUSED KERNELS ERROR: {str(e)}")

            # Check for common issues
            if "Permission denied" in str(e) or "cannot create directory" in str(e):
                print(f"FUSED KERNELS ERROR: This might be a file permission issue.")
                print(f"FUSED KERNELS ERROR: Check write permissions for: {buildpath}")
                print(f"FUSED KERNELS ERROR: You may need to delete the build directory and retry.")
            elif "file is locked" in str(e) or "resource temporarily unavailable" in str(e):
                print(f"FUSED KERNELS ERROR: Files appear to be locked by another process.")
                print(f"FUSED KERNELS ERROR: Make sure no other training processes are running.")
                print(f"FUSED KERNELS ERROR: Try deleting {buildpath} and retrying.")
            elif "nvcc not found" in str(e) or "CUDA_HOME" in str(e):
                print(f"FUSED KERNELS ERROR: CUDA installation issue detected.")
                print(f"FUSED KERNELS ERROR: Make sure CUDA is properly installed and CUDA_HOME is set.")

            print(f"FUSED KERNELS ERROR: Full build directory path: {buildpath}")
            raise

    # ==============
    # Fused softmax.
    # ==============

    if torch.version.hip is not None:
        extra_include_paths = [os.path.abspath(srcpath)]
    else:
        extra_include_paths = []

    if torch.version.hip is not None:
        extra_cuda_flags = [
            "-D__HIP_NO_HALF_OPERATORS__=1",
            "-D__HIP_NO_HALF_CONVERSIONS__=1",
        ]
    else:
        extra_cuda_flags = [
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ]

    print("\nFUSED KERNELS: Building/loading 3 fused kernels...")
    print("-"*60)

    # Upper triangular softmax.
    print("\n[1/3] Building scaled_upper_triang_masked_softmax_cuda...")
    sources = [
        srcpath / "scaled_upper_triang_masked_softmax.cpp",
        srcpath / "scaled_upper_triang_masked_softmax_cuda.cu",
    ]
    scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
        "scaled_upper_triang_masked_softmax_cuda",
        sources,
        extra_cuda_flags,
        extra_include_paths,
    )

    # Masked softmax.
    print("\n[2/3] Building scaled_masked_softmax_cuda...")
    sources = [
        srcpath / "scaled_masked_softmax.cpp",
        srcpath / "scaled_masked_softmax_cuda.cu",
    ]
    scaled_masked_softmax_cuda = _cpp_extention_load_helper(
        "scaled_masked_softmax_cuda", sources, extra_cuda_flags, extra_include_paths
    )

    # fused rope
    print("\n[3/3] Building fused_rotary_positional_embedding...")
    sources = [
        srcpath / "fused_rotary_positional_embedding.cpp",
        srcpath / "fused_rotary_positional_embedding_cuda.cu",
    ]
    fused_rotary_positional_embedding = _cpp_extention_load_helper(
        "fused_rotary_positional_embedding",
        sources,
        extra_cuda_flags,
        extra_include_paths,
    )

    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"FUSED KERNELS: All kernels loaded successfully!")
    print(f"FUSED KERNELS: Total time: {total_time:.2f} seconds")
    print("="*80 + "\n")
    sys.stdout.flush()


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


def _create_build_dir(buildpath):
    try:
        if not os.path.exists(buildpath):
            print(f"FUSED KERNELS: Creating build directory: {buildpath}")
            os.mkdir(buildpath)
            print(f"FUSED KERNELS: Build directory created successfully")
        else:
            print(f"FUSED KERNELS: Build directory already exists: {buildpath}")
            # Check if we have write permissions
            test_file = buildpath / ".write_test"
            try:
                test_file.touch()
                test_file.unlink()
                print(f"FUSED KERNELS: Build directory is writable")
            except Exception as e:
                print(f"FUSED KERNELS WARNING: Build directory may not be writable: {e}")
    except OSError as e:
        print(f"FUSED KERNELS ERROR: Failed to create build directory {buildpath}")
        print(f"FUSED KERNELS ERROR: {str(e)}")
        if "Permission denied" in str(e):
            print(f"FUSED KERNELS ERROR: Check file permissions for parent directory")
        raise


def load_fused_kernels():
    print("FUSED KERNELS: Checking if fused kernels are available...")
    try:
        import scaled_upper_triang_masked_softmax_cuda
        print("FUSED KERNELS: ✓ scaled_upper_triang_masked_softmax_cuda imported successfully")

        import scaled_masked_softmax_cuda
        print("FUSED KERNELS: ✓ scaled_masked_softmax_cuda imported successfully")

        import fused_rotary_positional_embedding
        print("FUSED KERNELS: ✓ fused_rotary_positional_embedding imported successfully")

        print("FUSED KERNELS: All fused kernels are available and ready to use!")

    except (ImportError, ModuleNotFoundError) as e:
        print("\n" + "!"*100)
        print("FUSED KERNELS ERROR: Failed to import fused kernels!")
        print(f"FUSED KERNELS ERROR: {str(e)}")
        print("!"*100)
        print("\nFUSED KERNELS: Fused kernels are not built yet.")
        print("FUSED KERNELS: To build them, run the following in Python:")
        print("\n    from megatron.fused_kernels import load")
        print("    load()")
        print("\nFUSED KERNELS: This will take 30-60 seconds on first run.")
        print("FUSED KERNELS: Once built, they will be cached for future runs.")
        print("!"*100 + "\n")
        exit()
    return
