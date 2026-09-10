#! /bin/bash
#SBATCH --job-name=slurm_torchrun_job
#SBATCH --output=slurm_torchrun_job-%j.out
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=2
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000                
#SBATCH --no-requeue

# --- Load necessary modules (might not be required depending on your HPC environment) ---
module load python-waterboa apptainer gcc openmpi ...

# --- Set up directories and container image ---
export SINGULARITY_TMPDIR="Artifacts/TEMP"
export CONTAINER_IMAGE=Artifacts/image.sif

# --- Set up distributed environment variables ---
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=30001
export COUNT_NODE=$SLURM_NNODES

echo "JOB ID: $SLURM_JOBID"
echo "NODES: $COUNT_NODE"
echo "HOSTNAMES: $HOSTNAMES"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# --- Create DeepSpeed hostfile ---
# This script should generate the hostfile using the SLURM_JOBID
bash write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile
export DLTS_HOSTFILE=hostfiles/hosts_$SLURM_JOBID

# --- Set WANDB API Key ---
export WANDB_API_KEY=""

# --- Execute the distributed training job using srun ---
srun -l apptainer exec --nv --bind /:/ $CONTAINER_IMAGE \
  bash -c '
    set -ex # Exit on error and print commands
    # --- Environment setup inside the container ---
    # Optional: Create a unique cache directory for each job run
    export TRITON_CACHE_DIR="/tmp/TRITON_TEMP_$SLURM_JOBID"
    mkdir -p $TRITON_CACHE_DIR
    export OMP_NUM_THREADS=10 # Might not be needed/might need adjustment based on your CPU setup
    # Map Slurm variables to standard distributed training variables
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    # --- Log environment for debugging ---
    echo "--------------------------------------------------"
    echo "Node ID: $SLURM_NODEID | Rank: $RANK | World Size: $WORLD_SIZE"
    echo "MASTER_ADDR: $MASTER_ADDR | MASTER_PORT: $MASTER_PORT"
    echo "DLTS_HOSTFILE: $DLTS_HOSTFILE"
    echo "WANDB API Key is set." # Avoid printing the key to logs
    echo "--------------------------------------------------"
    
    # Optional: Display installed packages
    # pip freeze --all
    
    # --- Run the training script ---
    cd /path_to_gpt_neox_codebase
    
    python deepy_torchrun.py train.py config.yml
'