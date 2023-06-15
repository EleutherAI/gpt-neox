source /gpfs/alpine/csc499/scratch/kshitijkg/setup.sh
source activate magma

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/kshitijkg/cache

TRAIN_PATH=/gpfs/alpine/csc499/scratch/kshitijkg/magma/gpt-neox
cd $TRAIN_PATH

bash /gpfs/alpine/csc499/scratch/kshitijkg/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/kshitijkg/hostfiles/$LSB_JOBID-hosts

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
	--conf_dir $TRAIN_PATH/configs magma_pythia_410M.yml magma_setup.yml 
