#!/bin/bash

# Directory containing log files
log_dir="logs"

# Function to parse and calculate averages
parse_logs() {
    local file=$1
    local total_flops=0
    local total_samples=0
    local flops_count=0
    local samples_count=0
    
    while IFS= read -r line
    do
        # Check if the line contains 'approx flops per GPU'
        if [[ $line =~ approx\ flops\ per\ GPU:\ ([0-9]+\.[0-9]+)TFLOPS ]]; then
            flops=${BASH_REMATCH[1]}
            total_flops=$(echo "$total_flops + $flops" | bc)
            flops_count=$((flops_count + 1))
        fi
        
        # Check if the line contains 'samples/sec'
        if [[ $line =~ samples/sec:\ ([0-9]+\.[0-9]+) ]]; then
            samples=${BASH_REMATCH[1]}
            total_samples=$(echo "$total_samples + $samples" | bc)
            samples_count=$((samples_count + 1))
        fi
    done < "$file"
    
    # Calculate averages
    if [ $flops_count -ne 0 ]; then
        avg_flops=$(echo "scale=2; $total_flops / $flops_count" | bc)
    else
        avg_flops="N/A"
    fi
    
    if [ $samples_count -ne 0 ]; then
        avg_samples=$(echo "scale=2; $total_samples / $samples_count" | bc)
    else
        avg_samples="N/A"
    fi
    
    echo "File: $file"
    echo "Average approx flops per GPU: $avg_flops TFLOPS"
    echo "Average samples/sec: $avg_samples"
    echo
}

# Loop through all files in the log directory
for log_file in "$log_dir"/*.out
do
    parse_logs "$log_file"
done
