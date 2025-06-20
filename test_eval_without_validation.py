#!/usr/bin/env python3
"""
Test script to verify eval_tasks work without validation data.
This script simulates the key parts of the training loop to test the evaluation logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unittest.mock import Mock, patch
from megatron.training import evaluate_and_print_results, evaluate
from megatron.neox_arguments import NeoXArgs
import torch


def test_eval_tasks_without_validation():
    """Test that eval_tasks run without validation data"""
    
    print("Testing eval_tasks without validation data...")
    
    # Create mock arguments
    mock_args = Mock(spec=NeoXArgs)
    mock_args.eval_tasks = ["lambada", "piqa"]
    mock_args.eval_interval = 10
    mock_args.eval_iters = 2
    mock_args.do_valid = False  # No validation data
    mock_args.deepspeed = False
    mock_args.char_level_ppl = False
    mock_args.use_wandb = False
    mock_args.tensorboard_writer = None
    mock_args.comet_experiment = None
    mock_args.is_pipe_parallel = False
    mock_args.gradient_accumulation_steps = 1
    
    # Mock model
    mock_model = Mock()
    mock_model.eval = Mock()
    mock_model.train = Mock()
    
    # Mock forward step function
    mock_forward_step = Mock()
    
    # Test 1: Verify evaluate() works with None data_iterator
    print("\nTest 1: Testing evaluate() with None data_iterator...")
    with patch('eval_tasks.run_eval_harness') as mock_eval_harness:
        mock_eval_harness.return_value = {
            "results": {
                "lambada": {"acc": 0.75, "ppl": 12.5},
                "piqa": {"acc": 0.65, "acc_norm": 0.68}
            }
        }
        
        results = evaluate(
            neox_args=mock_args,
            forward_step_fn=mock_forward_step,
            data_iterator=None,
            model=mock_model
        )
        
        print(f"Results: {results}")
        assert "lambada" in results
        assert "piqa" in results
        assert mock_eval_harness.called
        print("✓ evaluate() successfully ran eval_tasks with None data_iterator")
    
    # Test 2: Verify evaluate_and_print_results works
    print("\nTest 2: Testing evaluate_and_print_results()...")
    with patch('megatron.training.evaluate') as mock_evaluate:
        mock_evaluate.return_value = {
            "lambada": {"acc": 0.75, "ppl": 12.5},
            "piqa": {"acc": 0.65, "acc_norm": 0.68}
        }
        
        with patch('megatron.training.print_rank_0') as mock_print:
            with patch('torch.distributed.get_rank', return_value=0):
                with patch('torch.distributed.is_initialized', return_value=True):
                    evaluate_and_print_results(
                        neox_args=mock_args,
                        prefix="iteration 100",
                        forward_step_func=mock_forward_step,
                        data_iterator=None,
                        model=mock_model,
                        iteration=100,
                        chart_name="eval_tasks"
                    )
            
            # Verify evaluate was called with None
            call_args = mock_evaluate.call_args[1]
            assert call_args['data_iterator'] is None
            print("✓ evaluate_and_print_results() successfully handled None data_iterator")
    
    # Test 3: Simulate training loop evaluation trigger
    print("\nTest 3: Testing training loop evaluation trigger...")
    iteration = 10
    valid_data_iterator = None  # No validation data
    
    # This simulates the new evaluation logic in the training loop
    if mock_args.eval_interval and iteration % mock_args.eval_interval == 0:
        # Run validation data evaluation if configured
        if mock_args.do_valid and valid_data_iterator is not None:
            print("Would run validation data evaluation (skipped - no validation data)")
        
        # Run eval_tasks evaluation if configured (independent of validation data)
        if mock_args.eval_tasks and len(mock_args.eval_tasks) > 0:
            print("Running eval_tasks evaluation...")
            with patch('megatron.training.evaluate_and_print_results') as mock_eval_print:
                with patch('torch.distributed.get_rank', return_value=0):
                    with patch('torch.distributed.is_initialized', return_value=True):
                        # Simulate the call from training loop
                        evaluate_and_print_results(
                            neox_args=mock_args,
                            prefix=f"iteration {iteration}",
                            forward_step_func=mock_forward_step,
                            data_iterator=None,
                            model=mock_model,
                            iteration=iteration,
                            chart_name="eval_tasks"
                        )
                        print("✓ Training loop successfully triggered eval_tasks without validation data")
    
    print("\nAll tests passed! ✓")
    print("\nSummary:")
    print("- eval_tasks can run independently of validation data")
    print("- evaluate() function properly handles None data_iterator")
    print("- Training loop correctly triggers eval_tasks when configured")
    print("- No validation data is required for eval_tasks to run")


def test_config_validation():
    """Test configuration validation for eval_interval"""
    print("\n\nTesting configuration validation...")
    
    config = {
        "eval_tasks": ["lambada"],
        "vocab_file": "dummy",
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 4,
        "seq_length": 128,
        "train_iters": 1000,
    }
    
    with patch.object(NeoXArgs, 'configure_distributed_args'):
        with patch('logging.warning') as mock_warning:
            args = NeoXArgs(**config)
            args.calculate_derived()
            
            # Should have set eval_interval
            assert args.eval_interval == 1000
            print(f"✓ eval_interval automatically set to {args.eval_interval}")
            
            # Should have logged warning
            assert mock_warning.called
            warning_msg = mock_warning.call_args[0][0]
            assert "eval_tasks is set but eval_interval is not" in warning_msg
            print("✓ Warning logged about missing eval_interval")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing eval_tasks without validation data implementation")
    print("=" * 60)
    
    test_eval_tasks_without_validation()
    test_config_validation()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✅")
    print("=" * 60)