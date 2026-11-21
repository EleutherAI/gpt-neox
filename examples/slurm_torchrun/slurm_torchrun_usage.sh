#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8

# deepy_torchrun.py starts one torchrun per node, so run one task per node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# NeoX counts the GPUs in the hostfile to derive the batch size
scontrol show hostnames "$SLURM_JOB_NODELIST" | sed 's/$/ slots=8/' > hostfile_$SLURM_JOB_ID
export DLTS_HOSTFILE=hostfile_$SLURM_JOB_ID

srun python deepy_torchrun.py train.py configs/125M.yml configs/local_setup.yml
