#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --partition=gputest
#SBATCH --time=0-00:15:00
#SBATCH --gpus-per-node=v100:4
#SBATCH --account=project_2010225
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_ID.err logs/latest.err

CONFIG=350M.yml

# create hostfile
HOSTFILE="hostfiles/$SLURM_JOB_ID.txt"
mkdir -p $(dirname "$HOSTFILE")
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read n; do
    echo "$n slots=$SLURM_GPUS_ON_NODE" >> "$HOSTFILE"
done
    
echo "START: $(date)"

srun puhti_launch.sh run.py train.py "$CONFIG" --hostfile "$HOSTFILE"

echo "END: $(date)"
