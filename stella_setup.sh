#!/usr/bin/env bash

apt-get install zstd
pip install -r requirements.txt

# Install Apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git

# make sure I am using LibPython-dev version 3.8
echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' >> /etc/apt/sources.list && sudo apt update && sudo apt install libpython3.8-dev

# get example data
wget http://eaidata.bmk.sh/data/enron_emails.jsonl.zst
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

# unzip
zstd -d enron_emails.jsonl.zst

# process data
python tools/preprocess_data.py \
       --input enron_emails.jsonl \
       --output-prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file gpt2-merges.txt \
       --append-eod