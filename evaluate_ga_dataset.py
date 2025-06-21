#!/usr/bin/env python
"""
Evaluate model perplexity specifically on the gradient ascent dataset
to verify unlearning effectiveness.
"""

import sys
from megatron.neox_arguments import NeoXArgs
from megatron.training import setup_model_and_optimizer
from megatron.initialize import initialize_megatron
from megatron import print_rank_0
from eval import main as eval_main


def evaluate_ga_dataset(config_files):
    """
    Evaluate model on the GA dataset at different checkpoints
    to track unlearning progress.
    """
    # Load configuration
    neox_args = NeoXArgs.from_ymls(config_files)
    
    # Override data path to point to GA dataset for evaluation
    neox_args.data_path = neox_args.ga_dataset
    neox_args.split = "100,0,0"  # Use all data for evaluation
    
    # Set evaluation mode
    neox_args.do_train = False
    neox_args.do_valid = False
    neox_args.do_test = True
    
    # Run evaluation
    print_rank_0(f"\n{'='*50}")
    print_rank_0(f"Evaluating on GA dataset: {neox_args.ga_dataset}")
    print_rank_0(f"{'='*50}\n")
    
    eval_main(neox_args)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_ga_dataset.py config1.yml [config2.yml ...]")
        sys.exit(1)
    
    evaluate_ga_dataset(sys.argv[1:])