#!/bin/bash
# Basic for loop
names='meg1B_mp.yml meg2B_mp.yml meg4B_mp.yml'
for name in $names
do
echo "RUNNING ${name}"
./deepy.py pretrain_gpt2.py -d configs $name eleutherai_cluster.yml
done
echo "All done"
