#!/bin/bash
#SBATCH --job-name="neox-memorization"
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --time-min=1-12:00:00
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --comment=neox
#SBATCH --exclude=gpu-st-p4d-24xlarge-[269,281,284]
#SBATCH --exclusive

module load openmpi

#source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.4/efa/lib:/usr/local/cuda-11.4/lib:/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without these set; See https://github.com/NVIDIA/nccl/issues/676
#export NCCL_P2P_DISABLE=1
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME="eth0"
#export NCCL_DEBUG_SUBSYS=ALL

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_plm="^slurm"
# export I_MPI_PMI_LIBRARY="/opt/slurm/lib/libslurm.so"

TRAIN_PATH=/fsx/orz/gpt-neox-memorization

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json
export TORCH_EXTENSIONS_DIR=/fsx/orz/tmp/torch/
# # Add fused kernels to PYTHONPATH

cd $TRAIN_PATH

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
export MASTER_PORT=1000
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`


echo go $COUNT_NODE
which mpicc
python3 $TRAIN_PATH/deepy.py evaluation_script.py -d configs pythia-125M.yml sampling.yml

set +x