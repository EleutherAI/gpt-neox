#!/bin/bash
#... your SLURM arguments here
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8         
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --output=34b_replication_%j.out
#SBATCH --error=34b_replication_%j.out
#SBATCH --exclusive
#SBATCH --open-mode=append
#SBATCH --requeue

# setup the environment using the script we created before
source /fsx/proj-mathlm/conda_setup_deeperspeed.sh
#source /fsx/quentin/setup.sh

ds_report

# set distributed env variable flags such as NCCL_DEBUG here

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Move to the gpt-neox install
TRAIN_PATH=/path/to/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job here
# Should write to a hostfile that contains lines of format `<machine IP> slots=<NUM_GPUS_PER_NODE>`
bash /helper/script/write_hostfile.sh
export DLTS_HOSTFILE=path/to/hostfile/hosts_$SLURM_JOBID


# launch distributed job. If using `"deepspeed_slurm": true` and `"launcher": "slurm"` on a SLURM cluster, 
# then NeoX will handle the creation of a distributed run across 256 gpus.
python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
        --conf_dir /path/to/math-lm/pretraining llemma_34b.yml data_mixture.yml   