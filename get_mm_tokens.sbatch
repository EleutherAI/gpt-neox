#!/bin/bash
#SBATCH --job-name=most_memorized_sequences
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=6
#SBATCH --output=%x_%j.out
#SBATCH --comment=neox
#SBATCH --exclude=gpu-st-p4d-24xlarge-269

module load openmpi

TRAIN_PATH=/fsx/orz/gpt-neox-memorization
hostfile=$TRAIN_PATH/hosts

> $hostfile

for i in `scontrol show hostnames "$SLURM_JOB_NODELIST"`
do 
    echo $i slots=8 >> $hostfile
done

chmod 777 $hostfile
source /fsx/orz/miniconda3/bin/activate memorization

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=10245
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export FI_EFA_FORK_SAFE=1
python3 deepy.py get_mm_tokens.py get_mm_config.yml

set +x