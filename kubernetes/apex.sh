#!/usr/bin/env bash

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ -y
cd ..
rm -r apex

python prepare_data.py
mv gpt2-vocav.json data/gpt2-vocab.json
mv gpt2-merges.txt data/gpt2-merges.txt

git checkout sparse_attn
