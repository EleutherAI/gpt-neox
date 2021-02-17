#! /bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=127.0.0.1
MASTER_PORT=2000
NNODES=1
NODE_RANK=1

export DLWS_NUM_WORKER=${NNODES}
export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

# DATA OPTIONS: 
DATA_DIR="${DATA_DIR:-data}"
DATA_PATH=$DATA_DIR/enron/enron_text_document
VOCAB_PATH=$DATA_DIR/gpt2-vocab.json
MERGE_PATH=$DATA_DIR/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_large_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
#config_json="configs/deepspeed_configs/ds_zero_stage_2_config.json"
config_json="configs/deepspeed_configs/ds_zero_stage_1_config.json"
#config_json="configs/deepspeed_configs/ds_config.json"

# Training options: 
# Megatron Model Parallelism
mp_size=1
# DeepSpeed Pipeline parallelism
pp_size=2
# TOTAL BATCH SIZE = BATCHSIZE(pergpu) * GAS * N_GPUS
# ensure batch size details are consistent between here and the deepspeed config
BATCHSIZE=4 
GAS=16
LOGDIR="tensorboard_data/${NLAYERS}l_${NHIDDEN}h_${NNODES}n_${GPUS_PER_NODE}g_${pp_size}pp_${mp_size}mp_${BATCHSIZE}b_ds4"

#ZeRO Configs
stage=1
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Actication Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# GPT options:
NLAYERS=24
NHIDDEN=1536
NHEADS=16
SEQLEN=1024
LR="2.5e-4"
MINLR="2.5e-5"
WEIGHTDECAY=0
DROPOUT=0
SPARSITY='interspersed'
TRAIN_ITERS=320000

gpt_options=" \
        --model-parallel-size ${mp_size} \
        --pipe-parallel-size ${pp_size} \
        --num-layers $NLAYERS \
        --hidden-size $NHIDDEN \
        --num-attention-heads $NHEADS \
        --seq-length $SEQLEN \
        --max-position-embeddings 1024 \
        --batch-size $BATCHSIZE \
        --gas $GAS \
        --train-iters $TRAIN_ITERS \
        --lr-decay-iters $TRAIN_ITERS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr $LR \
        --lr-decay-style cosine \
        --min-lr $MINLR \
        --weight-decay $WEIGHTDECAY \
        --attention-dropout $DROPOUT \
        --hidden-dropout $DROPOUT \
        --clip-grad 1.0 \
        --warmup 0.01 \
        --checkpoint-activations \
        --log-interval 1 \
        --save-interval 500 \
        --eval-interval 100 \
        --eval-iters 10 \
        --fp16 \
        --tensorboard-dir ${LOGDIR} \
        --sparsity $SPARSITY
"
  
 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs} 
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--checkpoint-activations \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

full_options="${gpt_options} ${deepspeed_options} ${chkp_opt}"

DS_EXE="${DS_EXE:-deepspeed}"

run_cmd="$DS_EXE pretrain_gpt2.py $@ ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
