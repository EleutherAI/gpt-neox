#!/bin/bash
#SBATCH --job-name="neox-memorization"
#SBATCH --partition=compute-od-gpu
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=%x_%j.out  # Set this dir where you want slurm outs to go
#SBATCH --exclusive

module load openmpi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib:/opt/slurm/lib/libpmi.so
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export TORCH_EXTENSIONS_DIR=/home/orz/tmp/torch/
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export NCCL_DEBUG=info
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0
export OMPI_MCA_pml="^cm"
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
export OMPI_MCA_plm_rsh_no_tree_spawn=1
export I_MPI_PMI_LIBRARY="/opt/slurm/lib/libpmi.so"

TRAIN_PATH=/fsx/orz/gpt-neox

# Hide duplicated errors using this hack - will be properly fixed in pt-1.12
export TORCHELASTIC_ERROR_FILE=$TRAIN_PATH/tmp/torch-elastic-error.json

# Add fused kernels to PYTHONPATH
export PYTHONPATH="/fsx/orz/.local/lib64/python3.7/site-packages/fused_kernels-0.0.1-py3.7-linux-x86_64.egg"

cd $TRAIN_PATH

hostfile=/fsx/orz/gpt-neox/hosts
scontrol show hostnames "$SLURM_JOB_NODELIST"

> $hostfile

for i in `scontrol show hostnames "$SLURM_JOB_NODELIST"`
do 
    echo $i slots=8 >> $hostfile
done


python3 $TRAIN_PATH/deepy.py evaluation_script.py -d configs 6-7B.yml sampling.yml

set +x