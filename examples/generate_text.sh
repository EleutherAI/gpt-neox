#!/bin/bash

DATA_DIR="${DATA_DIR:-data}"
VOCAB_PATH=$DATA_DIR/gpt2-vocab.json
MERGE_PATH=$DATA_DIR/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

python tools/generate_samples_gpt2.py \
       --model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_PATH \
       --merge-file $MERGE_PATH \
       --genfile unconditional_samples.json \
       --num-samples 2 \
       --top_p 0.9 \
       --recompute
