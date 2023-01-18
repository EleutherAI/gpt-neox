#!/bin/bash
#SBATCH --job-name="memorization"
#SBATCH --partition=g40423
#SBATCH --nodes=32
#SBATCH --time-min=7-12:00:00
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --cpus-per-task=6
#SBATCH --output=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --comment=neox
#SBATCH --exclusive

module load openmpi
module load cuda/11.6


export GIT_DISCOVERY_ACROSS_FILESYSTEM=0
TRAIN_PATH=/fsx/orz/gpt-neox-memorization
cd /scratch/ # Test if scratch works

MODELS=(1.3b-deduped 2.7b-deduped)
CHECKPOINTS=(143000)
srun --partition g40423 --comment neox -n $SLURM_NPROCS /fsx/orz/memorization/bin/python3 $TRAIN_PATH/test_gpu.py
echo $TRAIN_PATH
for model in ${MODELS[@]}
do
for checkpoint in ${CHECKPOINTS[@]}
    do
    export MODEL=$model
    export CHECKPOINT=$checkpoint
    srun -n $SLURM_NPROCS --comment neox /fsx/orz/memorization/bin/python3 $TRAIN_PATH/eval_memorization_2.py
done
done
