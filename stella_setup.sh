#!/usr/bin/env bash

apt-get install zsdt
pip install -r requirements.txt
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git

wget http://eaidata.bmk.sh/data/enron_emails.jsonl.zst
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

zstd -d enron_emails.jsonl.zst

python tools/preprocess_data.py \
       --input enron_emails.jsonl \
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod
