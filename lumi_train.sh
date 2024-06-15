#!/bin/bash
#SBATCH --job-name=test_gpt_neox_LLama7B_8N_PP_4_10GAS_CPUBINDINGS
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
#SBATCH --gpus-per-node=8
#SBATCH --account=project_462000319
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# symlink logs/latest.out and logs/latest.err
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -f -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

module purge
module load LUMI
module load PyTorch/2.2.2-rocm-5.6.1-python-3.10-singularity-20240404
source lumi_config.sh
export PYTHONWARNINGS=ignore
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export TRANSFORMERS_CACHE=$HF_HOME

CONFIG="configs/llama/7B.yml configs/llama/train_config.yml configs/slurm_local.yml"

# create hostfile
rm hostfiles/*
HOSTFILE="hostfiles/$SLURM_JOB_ID.txt"
mkdir -p $(dirname "$HOSTFILE")
scontrol show hostnames "$SLURM_JOB_NODELIST" | while read n; do
    echo "$n slots=$SLURM_GPUS_ON_NODE" >> "$HOSTFILE"
done
c=fe
MYMASKS="0x${c}000000000000,0x${c}00000000000000,0x${c}0000,0x${c}000000,0x${c},0x${c}00,0x${c}00000000,0x${c}0000000000"

echo "START: $(date)"

srun --cpu-bind=mask_cpu:$MYMASKS --label lumi_launch.sh run.py train.py "$CONFIG" --hostfile "$HOSTFILE"

echo "END: $(date)"
