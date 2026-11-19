"""Install the Astral FlashAttention wheel for the image's PyTorch build."""

import re
import subprocess
import sys
from importlib.metadata import version

import torch


def main():
    torch_version = version("torch")
    cuda_version = torch.version.cuda
    stable_torch = re.fullmatch(r"(\d+\.\d+)\.\d+(?:\+[\w.]+)?", torch_version)
    if not stable_torch or not cuda_version:
        sys.exit(
            f"Astral FlashAttention requires a stable CUDA PyTorch build; found "
            f"PyTorch {torch_version} with CUDA {cuda_version}. Set PYTORCH_IMAGE "
            "to a base with a matching Astral wheel or disable USE_ASTRAL_FLASH_ATTENTION."
        )

    cuda = ".".join(cuda_version.split(".")[:2])
    index = f"https://wheels.astral.sh/simple/cu{cuda.replace('.', '')}/"
    wheel = f"flash-attn==2.8.3+cu.{cuda}.torch.{stable_torch.group(1)}"
    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--system",
            "--python",
            sys.executable,
            "--only-binary",
            ":all:",
            "--index",
            index,
            f"torch=={torch_version}",
            wheel,
        ]
    )
    if result.returncode:
        sys.exit(f"Could not install {wheel} from {index} for PyTorch {torch_version}.")

    # Check that the extension loads against the PyTorch and CUDA libraries.
    subprocess.run(
        [sys.executable, "-c", "import torch; import flash_attn"], check=True
    )


if __name__ == "__main__":
    main()
