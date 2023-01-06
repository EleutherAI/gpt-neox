#!/bin/bash
#SBATCH --job-name="tune"
#SBATCH --partition=a100-cu117
#SBATCH --mem-per-cpu=16GB        # Amount of CPU memory
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8      # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4          # Number of cores per tasks
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=%x_%j.out   # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out    # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing

source /opt/hpcx/hpcx-init.sh
hpcx_load

export PYTHONFAULTHANDLER=1

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json

source /mnt/nvme/home/$(whoami)/conda/bin/activate neox

# Move to the gpt-neox install
TRAIN_PATH=/mnt/nvme/home/$(whoami)/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
/mnt/nvme/home/$(whoami)/write_hostfile.sh

export DLTS_HOSTFILE=/mnt/nvme/home/$(whoami)/hostfiles/hosts_$SLURM_JOBID
echo $DLTS_HOSTFILE
python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
    --autotuning tune \
    --conf_dir configs tune_1-3B.json cw_pile.json

