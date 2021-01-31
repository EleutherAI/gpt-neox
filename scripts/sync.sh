#!/usr/bin/env bash

# Push files to all nodes
# Usage
# sync.sh file [file2..]

echo Number of files to upload: $#

for file in "$@"
do
    full_path=$(realpath $file)
    echo Uploading $full_path
    pdcp -f 1024 -R ssh -w ^/job/hosts $full_path $full_path
done
