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


srcpath = Path(__file__).parent.absolute()
cc_flag = []
_, bare_metal_major, _ = _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
if int(bare_metal_major) >= 11:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")

nvcc_flags = [
    "-O3",
    "-gencode",
    "arch=compute_70,code=sm_70",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]
cuda_ext_args = {"cxx": ["-O3"], "nvcc": nvcc_flags + cc_flag}
layernorm_cuda_args = {
    "cxx": ["-O3"],
    "nvcc": nvcc_flags + cc_flag + ["-maxrregcount=50"],
}
setup(
    name="fused_kernels",
    version="0.0.1",
    author="Sid Black & Alejandro Molina et al.",
    author_email="alejandro.molina@aleph-alpha.de",
    include_package_data=False,
    ext_modules=[
        CUDAExtension(
            "scaled_upper_triang_masked_softmax_cuda",
            [
                str(srcpath / "scaled_upper_triang_masked_softmax.cpp"),
                str(srcpath / "scaled_upper_triang_masked_softmax_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        ),
        CUDAExtension(
            "scaled_masked_softmax_cuda",
            [
                str(srcpath / "scaled_masked_softmax.cpp"),
                str(srcpath / "scaled_masked_softmax_cuda.cu"),
            ],
            extra_compile_args=cuda_ext_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
