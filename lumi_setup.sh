#!/bin/bash

SCRIPT_DIR=$(dirname "$0")

cd "$SCRIPT_DIR"
source lumi_config.sh

mkdir -p $PYTHONUSERBASE

CMD="
source /opt/miniconda3/bin/activate pytorch
pip install -r $SCRIPT_DIR/requirements/lumi_requirements.txt --user
$CXX -O3 -Wall -shared -std=c++11 -fPIC megatron/data/helpers.cpp -o megatron/data/helpers.cpython-310-x86_64-linux-gnu.so
echo 'Installation finished'
"

REAL_PWD="$(realpath "$PWD")"

singularity exec --pwd "$REAL_PWD" "$CONTAINER" bash -c "$CMD"
