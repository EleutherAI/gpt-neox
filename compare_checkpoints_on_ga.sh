#!/bin/bash
# Compare model performance on GA dataset across checkpoints

CONFIG_PATH=$1
CHECKPOINT_DIR=$2

if [ -z "$CONFIG_PATH" ] || [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: ./compare_checkpoints_on_ga.sh config.yml checkpoint_dir"
    exit 1
fi

echo "Evaluating checkpoints on GA dataset..."
echo "=================================="

# Find all global_step directories
for checkpoint in $(find $CHECKPOINT_DIR -name "global_step*" -type d | sort -V); do
    step=$(basename $checkpoint | sed 's/global_step//')
    echo -e "\nEvaluating checkpoint at step $step"
    echo "-----------------------------------"
    
    # Create temp config with this checkpoint
    temp_config="/tmp/ga_eval_step${step}.yml"
    cat > $temp_config << EOF
{
  "include": "$CONFIG_PATH",
  "load": "$checkpoint",
  "finetune": true,
  "iteration": $step,
  "do_train": false,
  "do_valid": false,
  "do_test": true,
  "data_path": "$(grep ga_dataset $CONFIG_PATH | cut -d'"' -f4)",
  "split": "100,0,0",
}
EOF
    
    # Run evaluation
    python deepy.py eval.py $temp_config --eval_tasks lm_perplexity
    
    # Clean up
    rm $temp_config
done

echo -e "\nEvaluation complete!"