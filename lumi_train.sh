#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --partition=dev-g
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000558
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

source lumi_config.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

CONFIG=350M.yml

# create hostfile
HOSTFILE="hostfiles/$SLURM_JOB_ID.txt"
mkdir -p $(dirname "$HOSTFILE")
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read n; do
    echo "$n slots=$SLURM_GPUS_ON_NODE" >> "$HOSTFILE"
done

REAL_PWD="$(realpath "$PWD")"

echo "START: $(date)"

srun --label \
  singularity exec \
     --pwd "$REAL_PWD" \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /usr/lib64/libjansson.so.4 \
    -B "$SING_BIND" \
    "$CONTAINER" \
    ./lumi_launch.sh run.py train.py "$CONFIG" --hostfile "$HOSTFILE"

echo "END: $(date)"
