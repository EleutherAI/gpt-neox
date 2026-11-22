# GPT-NeoX Container Image Guide

This guide covers the Docker images and the native Apptainer definition, plus
conversion from Docker to Apptainer. Run the build commands from the
`gpt-neox` repository root so the recipes can access the requirements and
fused kernel sources.

```bash
cd gpt-neox
```

The recipes use uv to install packages from public PyPI into the base image's
system Python environment. They pass the NGC package constraint file to uv
when the base image provides one.

Build the standard Docker or Apptainer image from the repository root.

```bash
docker build -f containers/docker/Dockerfile -t gpt-neox:latest .
apptainer build gpt-neox.sif containers/apptainer/gpt-neox.def
```

## Image Matrix

| Image | Dockerfile | Base image | CUDA architecture | Use case |
| --- | --- | --- | --- | --- |
| `gpt-neox:latest` | `containers/docker/Dockerfile` | `nvcr.io/nvidia/pytorch:24.02-py3` | Base defaults | Standard GPT-NeoX image. |
| `gpt-neox:te` | `containers/docker/Dockerfile.TE` | `nvcr.io/nvidia/pytorch:24.02-py3` | `9.0` | Transformer Engine and FlashAttention on Hopper-class GPUs such as H100. |
| `gpt-neox:b200` | `containers/docker/Dockerfile.B200` | `nvcr.io/nvidia/pytorch:25.04-py3` | `10.0` | B200 systems that require the newer PyTorch and CUDA stack from NGC 25.04. |

## TE Image

Use `containers/docker/Dockerfile.TE` for the Transformer Engine image built on
the NGC PyTorch 24.02 base.

The image:

- installs `requirements.txt` and `requirements-onebitadam.txt`
- installs W&B from `requirements-wandb.txt`, shared by all images
- installs `transformer-engine[pytorch]==1.12`
- installs `flash-attn==2.5.6`
- pins `protobuf==4.25.8` to satisfy the cuDF requirement in NGC 24.02
- sets `TORCH_CUDA_ARCH_LIST=9.0`
- builds the GPT-NeoX fused kernels from `megatron/fused_kernels/`

In the TE image, the FlashAttention setup script tries to
download a matching upstream wheel. If none is available, it compiles against
the PyTorch in the base image.

Build the image:

```bash
docker build -f containers/docker/Dockerfile.TE -t gpt-neox:te .
```

Verify the image:

```bash
docker run --rm --gpus all --ipc=host gpt-neox:te nvidia-smi
docker run --rm --gpus all --ipc=host gpt-neox:te \
  python -c "import torch, wandb, flash_attn, transformer_engine; print(torch.__version__)"
```

## B200 Image

Use `containers/docker/Dockerfile.B200` for B200 systems. B200 support requires
a newer PyTorch and CUDA stack, so this image moves to the NGC PyTorch 25.04
base.

The B200 image runs as `mchorse`. It uses UID 1000 when it is free and
allocates an available UID when the base image already has a user at 1000.
For bind-mount permissions, check the resulting UID with
`docker run --rm gpt-neox:b200 id -u`.

The image:

- installs `requirements.txt` and `requirements-onebitadam.txt`
- installs W&B and Protobuf from `requirements-wandb.txt`
- sets `TORCH_CUDA_ARCH_LIST=10.0`
- relies on the NGC 25.04 base image for Transformer Engine
- rewrites `jinja2==3.1.4` to `jinja2==3.1.6` during the build to match the
  NGC base image's package constraint
- updates the base image's Protobuf constraint to match `requirements-wandb.txt`
- pins `fsspec==2024.6.1` and `dill==0.3.8` for compatibility with Datasets 2.x
- removes `chardet` after installation to avoid the `requests` dependency
  warning emitted during W&B imports in this base image
- builds the GPT-NeoX fused kernels from `megatron/fused_kernels/`

Build the image:

```bash
docker build -f containers/docker/Dockerfile.B200 -t gpt-neox:b200 .
```

Verify the image:

```bash
docker run --rm --gpus all --ipc=host gpt-neox:b200 nvidia-smi
docker run --rm --gpus all --ipc=host gpt-neox:b200 \
  python -c "import torch, wandb, transformer_engine; print(torch.__version__)"
```

## Skipping the GPT-NeoX Fused Kernel Build

The Docker and Apptainer images build the GPT-NeoX
fused kernels by default. Pass `--build-arg BUILD_FUSED_KERNELS=false` to skip
this step.

```bash
docker build --build-arg BUILD_FUSED_KERNELS=false \
  -f containers/docker/Dockerfile -t gpt-neox:no-fused-kernels .
```

With Apptainer 1.2 or later, the definition copies its build inputs from
the checkout.

```bash
apptainer build --build-arg BUILD_FUSED_KERNELS=false \
  gpt-neox-no-fused-kernels.sif containers/apptainer/gpt-neox.def
```

The flag controls GPT-NeoX's own extensions. The Transformer Engine image
installs FlashAttention with either setting. To avoid compiling the GPT-NeoX
extensions at runtime, set `scaled_upper_triang_masked_softmax_fusion`,
`scaled_masked_softmax_fusion`, and `rope_fusion` to `false` in the model config.

## Running a Container

Set the host paths that should be mounted into the container:

```bash
export NEOX_DATA_PATH=/path/to/data
export NEOX_CHECKPOINT_PATH=/path/to/checkpoints
export WANDB_API_KEY=your_wandb_key
```

Start an interactive shell:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v $NEOX_DATA_PATH:/home/mchorse/data \
  -v $NEOX_CHECKPOINT_PATH:/home/mchorse/chk \
  -v $(pwd):/home/mchorse/gpt-neox \
  gpt-neox:te \
  bash
```

Use `gpt-neox:b200` instead of `gpt-neox:te` when running the B200 image.

## Apptainer Equivalents

If your HPC environment requires Apptainer, convert the Docker images into
`.sif` files. When Docker is available on the build host, convert directly from
the local Docker daemon:

```bash
apptainer build containers/apptainer/gpt-neox-te-from-docker.sif docker-daemon://gpt-neox:te
apptainer build containers/apptainer/gpt-neox-b200-from-docker.sif docker-daemon://gpt-neox:b200
```

If Docker is not available on the target system, push the Docker images to a
registry from a Docker-capable host and build from the registry instead:

```bash
apptainer build gpt-neox-te.sif docker://your-registry/gpt-neox:te
apptainer build gpt-neox-b200.sif docker://your-registry/gpt-neox:b200
```

Verify the Apptainer images with `--nv` to enable NVIDIA GPU support:

```bash
apptainer exec --nv containers/apptainer/gpt-neox-te-from-docker.sif \
  python -c "import torch, wandb, flash_attn, transformer_engine; print(torch.__version__)"
apptainer exec --nv containers/apptainer/gpt-neox-b200-from-docker.sif \
  python -c "import torch, wandb, transformer_engine; print(torch.__version__)"
```

For `.sif` files produced from registry images, use the same
`apptainer exec --nv ...` pattern with the generated filenames.
