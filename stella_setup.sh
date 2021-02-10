#!/usr/bin/env bash

apt-get install zstd
pip install -r requirements.txt

# Install Apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git

# make sure I am using LibPython-dev version 3.8
echo 'deb http://archive.ubuntu.com/ubuntu/ focal main restricted' >> /etc/apt/sources.list && sudo apt update && sudo apt install libpython3.8-dev && apt-get install --upgrade libpython3-dev
