#!/bin/bash

# Script to convert NeoX checkpoints to HuggingFace format
# Usage: ./convert_checkpoint.sh <experiment_name> <checkpoint_number>
# Example: ./convert_checkpoint.sh annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <experiment_name> <checkpoint_number>"
    echo "Example: $0 annealing_filtered_v5_replace_with_escelations_wmdp_deep_fry_20x_upsampled 12653"
    exit 1
fi

EXPERIMENT_NAME=$1
CHECKPOINT_NUMBER=$2

# Define paths
INPUT_DIR="/checkpoints/${EXPERIMENT_NAME}/global_step${CHECKPOINT_NUMBER}"
OUTPUT_DIR="/checkpoints/hf_converted/${EXPERIMENT_NAME}/global_step${CHECKPOINT_NUMBER}"
CONFIGS_DIR="${INPUT_DIR}/configs"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if configs directory exists
if [ ! -d "$CONFIGS_DIR" ]; then
    echo "Error: Configs directory does not exist: $CONFIGS_DIR"
    exit 1
fi

# Find the config file (should be the only .yml file in configs dir)
CONFIG_FILE=$(find "$CONFIGS_DIR" -name "*.yml" | head -1)

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: No config file found in $CONFIGS_DIR"
    exit 1
fi

echo "Found config file: $CONFIG_FILE"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Converting checkpoint..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_FILE"

# Run the conversion
python tools/ckpts/convert_neox_to_hf.py \
    --input_dir="$INPUT_DIR" \
    --config_file="$CONFIG_FILE" \
    --output_dir="$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "Conversion completed successfully!"
    echo "Converted model saved to: $OUTPUT_DIR"
else
    echo "Conversion failed!"
    exit 1
fi