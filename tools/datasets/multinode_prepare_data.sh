#!/bin/bash

# USAGE:
# This script allows you to prepare your dataset using multiple nodes by chunking the individual files and distributed the chunks
# over the processes.
# This bash script takes a single text file as input argument.
# The text file contains a valid filepath in each line, leading to a jsonl-file.
# Furthermore an environment variable for the rank and the world size needs to be set.
# These default to the SLURM and OMPI variables in this order of priority, but they can be set manually as well
# using the variables $RANK and $WORLD_SIZE, which will overwrite the cluster-specific variables.
# You can also add all arguments of the prepare_data.py script to this script and it will simply pass them through.

# Parse command-line arguments
text_file="$1"
rank="${RANK:-${SLURM_PROCID:-$OMPI_COMM_WORLD_RANK}}"
world_size="${WORLD_SIZE:-${SLURM_NTASKS:-$OMPI_COMM_WORLD_SIZE}}"
num_lines=$(wc -l < "$text_file")
chunk_size=$((num_lines / world_size))
start_line=$((rank * chunk_size + 1))
end_line=$((start_line + chunk_size - 1))

# Make sure the last chunk includes all remaining lines
if [[ $rank == $((world_size - 1)) ]]; then
    end_line=$num_lines
fi

# Select the chunk of the text file that corresponds to the rank
chunk_file="chunk_${rank}.txt"
sed -n "${start_line},${end_line}p" "$text_file" > "$chunk_file"

# Parse additional flags to be passed to the Python script
shift 1  # Shift past the first three arguments
py_args=""
prefix_arg=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-prefix=*) prefix_arg="$1"; shift;;
        --output-prefix) prefix_arg="$1 $2"; shift 2;;
        --*) py_args="$py_args $1 $2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Add the rank to the --output-prefix argument if it is set
if [[ -n "$prefix_arg" ]]; then
    py_args="$py_args $prefix_arg$rank"
else
    # Inject a default --output-prefix argument containing the rank
    py_args="$py_args --output-prefix rank${rank}"
fi


echo "processing $chunk_file with rank $rank at world size $world_size"
echo "using the following args: $py_args"
# Call the Python script with the list of file paths in the chunk
python tools/datasets/preprocess_data.py --input $(tr '\n' ',' < "$chunk_file" | sed 's/,$/\n/') $py_args

# Clean up
rm "$chunk_file"
