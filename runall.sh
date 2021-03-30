#!/bin/bash
# Basic for loop
names='meg1B.yml meg2B.yml meg4B.yml meg6B.yml meg8B.yml'
for name in $names
do
echo "RUNNING ${name}"
./deepy.py pretrain_gpt2.py -d configs $name eleutherai_cluster.yml
done
echo "All done"
