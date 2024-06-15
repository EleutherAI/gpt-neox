#!/bin/bash

export PYTHONUSERBASE="pythonuserbase" 

export CC=gcc-10
export CXX=g++-10
export CPATH="$CPATH:/opt/miniconda3/include/python3.10:$PYTHONUSERBASE/lib/python3.10/site-packages/pybind11/include"

export CONTAINER="/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.2.0.sif"

#export SINGULARITY_BIND="$REAL_PWD"
