#!/usr/bin/env python3
"""Test script to verify gradient signs work with pipeline parallelism"""

import torch
import numpy as np
from megatron.data.data_utils import auto_detect_gradient_signs_path
from megatron.model.gpt2_model import cross_entropy_with_gradient_tracking
from megatron import print_rank_0

class MockNeoXArgs:
    def __init__(self):
        self._current_gradient_signs = None
        self._additional_losses = {}
        self.train_gradient_signs_data_paths = ["test"]
        self.is_pipe_parallel = True
        self.fp16_lm_cross_entropy = False

def test_gradient_tracking():
    """Test the gradient tracking loss function"""
    print_rank_0("Testing gradient tracking loss function...")
    
    # Create mock data
    batch_size = 4
    seq_len = 128
    vocab_size = 50304
    
    # Mock outputs and labels
    output = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_mask = torch.ones(batch_size, seq_len)
    
    # Create gradient signs: 2 ascent, 2 descent samples
    gradient_signs = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    gradient_signs = gradient_signs.unsqueeze(1).expand(batch_size, seq_len)
    
    # Create neox_args mock
    neox_args = MockNeoXArgs()
    neox_args._current_gradient_signs = gradient_signs
    
    # Test the loss function
    loss = cross_entropy_with_gradient_tracking(
        output, 
        (labels, loss_mask),
        _fp16=False,
        neox_args=neox_args
    )
    
    print_rank_0(f"Loss computed: {loss.item()}")
    print_rank_0(f"Additional losses: {neox_args._additional_losses}")
    
    # Verify results
    assert "gradient_ascent_loss" in neox_args._additional_losses
    assert "gradient_descent_loss" in neox_args._additional_losses
    assert "gradient_ascent_samples" in neox_args._additional_losses
    assert "gradient_descent_samples" in neox_args._additional_losses
    
    assert neox_args._additional_losses["gradient_ascent_samples"].item() == 2
    assert neox_args._additional_losses["gradient_descent_samples"].item() == 2
    
    print_rank_0("✓ Gradient tracking test passed!")

def test_auto_detect():
    """Test auto-detection of gradient signs files"""
    print_rank_0("\nTesting auto-detection of gradient signs files...")
    
    # Test with various prefixes
    test_cases = [
        "/path/to/data_text_document",
        "/path/to/data",
        None
    ]
    
    for prefix in test_cases:
        result = auto_detect_gradient_signs_path(prefix)
        print_rank_0(f"  Input: {prefix} -> Output: {result}")
    
    print_rank_0("✓ Auto-detection test completed!")

if __name__ == "__main__":
    # Initialize minimal torch distributed for print_rank_0
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29500',
            world_size=1,
            rank=0
        )
    
    # Initialize model parallel groups for mpu
    from megatron import mpu
    mpu.initialize_model_parallel(1)
    
    print_rank_0("=" * 60)
    print_rank_0("Testing Gradient Signs Implementation for Pipeline Parallel")
    print_rank_0("=" * 60)
    
    test_gradient_tracking()
    test_auto_detect()
    
    print_rank_0("\n✓ All tests passed!")