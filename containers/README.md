# GPT-NeoX Container Image Guide

This guide covers the custom Docker images in `containers/docker/` and the
matching Apptainer conversion workflow. Build these images from the
`gpt-neox` repository root, not from `containers/docker`, because the
Dockerfiles copy `requirements/*` and `megatron/fused_kernels/` from the build
context.

```bash
cd gpt-neox
```

## Image Matrix

| Image | Dockerfile | Base image | CUDA architecture | Use case |
| --- | --- | --- | --- | --- |
| `gpt-neox:te` | `containers/docker/Dockerfile.TE` | `nvcr.io/nvidia/pytorch:24.02-py3` | `9.0` | Transformer Engine and FlashAttention on Hopper-class GPUs such as H100. |
| `gpt-neox:b200` | `containers/docker/Dockerfile.B200` | `nvcr.io/nvidia/pytorch:25.04-py3` | `10.0` | B200 systems that require the newer PyTorch and CUDA stack from NGC 25.04. |

## TE Image

Use `containers/docker/Dockerfile.TE` for the Transformer Engine image built on
the NGC PyTorch 24.02 base.

The image:

- installs `requirements.txt` and `requirements-onebitadam.txt`
- installs `wandb==0.16.6`
- installs `transformer-engine[pytorch]==1.12`
- builds `flash-attn==2.5.6` from source
- pins `protobuf==3.20.*`
- sets `TORCH_CUDA_ARCH_LIST=9.0`
- builds the GPT-NeoX fused kernels from `megatron/fused_kernels/`

FlashAttention is built from source instead of installed from the wheel path in
`requirements-transformerengine.txt`, because prebuilt wheels are more likely
to hit ABI mismatches on the older NGC 24.02 PyTorch base. W&B is pinned to
`0.16.6` so it stays compatible with the `protobuf==3.20.*` pin.

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

The image:

- installs `requirements.txt` and `requirements-onebitadam.txt`
- installs `wandb==0.16.6`
- sets `TORCH_CUDA_ARCH_LIST=10.0`
- relies on the NGC 25.04 base image for Transformer Engine
- rewrites `jinja2==3.1.4` to `jinja2==3.1.6` during the build to match the
  NGC base-image pip constraint
- keeps the base image's newer protobuf constraint instead of forcing
  `protobuf==3.20.*`
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
