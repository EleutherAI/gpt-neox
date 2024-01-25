#!/bin/bash
#SBATCH --job-name=convert_mistral_7B
#SBATCH --partition=a40x            # Make sure you need this
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=convert_mistral_7B_%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=convert_mistral_7B_%x_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --time=0-00:12:00


# set up the environment using a setup script
source ~/setup.sh

ds_report

export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json
export TORCH_EXTENSIONS_DIR=./extensions/

# Move to the gpt-neox install
TRAIN_PATH=/weka/$(whoami)/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
export MASTER_ADDR=$(echo $MASTER_ADDR | cut -d '-' -f 2- | tr '-' '.')
bash ~/write_ip_hostfile.sh
export DLTS_HOSTFILE=/weka/$(whoami)/hostfiles/hosts_$SLURM_JOBID



python ./deepy.py convert_hf_to_sequential_mistral.py \
    -d configs mistral_7b.yml

# python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
#         --conf_dir configs/ mistral_7b.yml