#!/usr/bin/env bash

# Push files to all nodes
# Usage
# sync.sh file [file2..]

echo Number of files to upload: $#

for file in "$@"
do
    full_path=$(realpath $file)
    parentdir="$(dirname "$full_path")"
    echo Uploading $full_path to $parentdir
    pdcp -f 1024 -R ssh -w ^/job/hosts -r $full_path $parentdir
done
