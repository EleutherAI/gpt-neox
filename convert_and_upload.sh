#!/bin/bash
"""
Wrapper script for convert_and_upload.py

Usage:
    ./convert_and_upload.sh <experiment_name> <checkpoint_number> <hf_repo_name> [options]

Example:
    ./convert_and_upload.sh annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653 Unlearning/pythia1.5_modernbert_filtered_5percent_wmdp_deep_fry_20x_upsampled
"""

# Change to script directory
cd "$(dirname "$0")"

# Run the Python script with all arguments
python convert_and_upload.py "$@"