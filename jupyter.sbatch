#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=6
#SBATCH --output=%x_%j.out
#SBATCH --comment=neox

scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1
source /fsx/orz/miniconda3/bin/activate memorization
jupyter-notebook --ip=0.0.0.0 --port=8888
