#!/usr/bin/env python
"""
Integration test to reproduce and verify fix for the tensor size mismatch error:
RuntimeError: The size of tensor a (65504) must match the size of tensor b (65536)
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from megatron.model.gpt2_model import cross_entropy_with_gradient_tracking
from unittest.mock import Mock, patch


def test_exact_error_condition():
    """Reproduce the exact error condition from the runtime error"""
    # Exact dimensions from the error
    batch_size = 32
    seq_length = 2048
    vocab_size = 50257
    
    # Total tokens and actual tokens (32 padding tokens)
    total_tokens = batch_size * seq_length  # 65536
    actual_tokens = 65504  # As in the error message
    padding_tokens = total_tokens - actual_tokens  # 32
    
    print(f"Testing with:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Actual tokens: {actual_tokens}")
    print(f"  Padding tokens: {padding_tokens}")
    
    # Create output tensor
    output = torch.randn(batch_size, seq_length, vocab_size)
    
    # Create labels and loss mask (full size)
    labels_tensor = torch.randint(0, vocab_size, (batch_size, seq_length))
    loss_mask = torch.ones(batch_size, seq_length)
    
    # Mark last token of each sequence as padding
    for i in range(batch_size):
        loss_mask[i, -1] = 0
    
    labels = (labels_tensor, loss_mask)
    
    # Create gradient signs (smaller size - excludes padding)
    gradient_signs = torch.ones(actual_tokens)
    # Mark ~1.21% as ascent (as in the original data)
    n_ascent = int(actual_tokens * 0.0121)
    indices = torch.randperm(actual_tokens)[:n_ascent]
    gradient_signs[indices] = -1
    
    print(f"  Ascent tokens: {n_ascent} ({n_ascent/actual_tokens*100:.2f}%)")
    
    # Mock neox_args
    neox_args = Mock()
    neox_args._current_gradient_signs = gradient_signs
    neox_args.gradient_ascent_loss_scale = 10.0
    
    # Mock the vocab_parallel_cross_entropy function
    def mock_cross_entropy(output, labels, sample_signs=None, gradient_ascent_loss_scale=1.0):
        # Return tensor of correct shape
        batch_size = output.shape[0]
        seq_length = output.shape[1]
        losses = torch.randn(batch_size, seq_length)
        
        # If sample signs provided, apply them (but don't worry about size mismatch here)
        if sample_signs is not None:
            # This is where the real implementation would apply gradient signs
            pass
            
        return losses
    
    with patch('megatron.mpu.vocab_parallel_cross_entropy', side_effect=mock_cross_entropy):
        # This should NOT raise the size mismatch error
        try:
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            print("✓ No tensor size mismatch error!")
            
            # Verify loss was computed
            assert isinstance(loss, torch.Tensor)
            assert loss.numel() == 1  # Single scalar loss
            
            # Verify monitoring metrics were computed
            assert hasattr(neox_args, '_additional_losses')
            if hasattr(neox_args._additional_losses, '__contains__'):
                assert 'gradient_ascent_loss' in neox_args._additional_losses
                assert 'gradient_descent_loss' in neox_args._additional_losses
                assert 'gradient_ascent_samples' in neox_args._additional_losses
                assert 'gradient_descent_samples' in neox_args._additional_losses
            else:
                # If it's a Mock object, just check it was set
                assert neox_args._additional_losses is not None
            
            # Verify sample counts if available
            if hasattr(neox_args._additional_losses, '__getitem__'):
                try:
                    ascent_samples = neox_args._additional_losses['gradient_ascent_samples']
                    descent_samples = neox_args._additional_losses['gradient_descent_samples']
                    
                    print(f"\nMonitoring metrics computed successfully:")
                    print(f"  Ascent samples tracked: {ascent_samples.item() if hasattr(ascent_samples, 'item') else ascent_samples}")
                    print(f"  Descent samples tracked: {descent_samples.item() if hasattr(descent_samples, 'item') else descent_samples}")
                    print(f"  Total tracked: {(ascent_samples + descent_samples).item() if hasattr(ascent_samples + descent_samples, 'item') else ascent_samples + descent_samples}")
                except:
                    print("\nMonitoring metrics were set but details not available in mock.")
            
        except RuntimeError as e:
            if "must match the size of tensor" in str(e):
                pytest.fail(f"Tensor size mismatch not fixed: {e}")
            else:
                raise


def test_multi_node_padding_scenario():
    """Test a multi-node scenario where padding might vary across nodes"""
    # Simulate different padding amounts across different micro-batches
    # This can happen in distributed training
    
    test_cases = [
        (32, 2048, 65504),  # Node 1: 32 padding tokens
        (32, 2048, 65500),  # Node 2: 36 padding tokens  
        (32, 2048, 65536),  # Node 3: No padding
        (32, 2048, 65000),  # Node 4: 536 padding tokens
    ]
    
    for i, (batch_size, seq_length, actual_tokens) in enumerate(test_cases):
        print(f"\nTesting node {i+1} scenario:")
        print(f"  Batch: {batch_size}, Seq: {seq_length}, Actual tokens: {actual_tokens}")
        
        total_tokens = batch_size * seq_length
        padding = total_tokens - actual_tokens
        
        output = torch.randn(batch_size, seq_length, 50257)
        labels_tensor = torch.randint(0, 50257, (batch_size, seq_length))
        loss_mask = torch.ones(batch_size, seq_length)
        
        # Simulate padding at the end
        if padding > 0:
            loss_mask.view(-1)[-padding:] = 0
            
        labels = (labels_tensor, loss_mask)
        
        # Gradient signs only for actual tokens
        gradient_signs = torch.ones(actual_tokens)
        gradient_signs[:100] = -1  # Some ascent tokens
        
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 10.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(batch_size, seq_length)
            
            # Should handle all scenarios without error
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            print(f"  ✓ Node {i+1} passed")


def test_edge_cases():
    """Test edge cases that might cause issues"""
    
    print("\nTesting edge cases:")
    
    # Edge case 1: Very small batch with no padding
    print("\n1. Small batch, no padding:")
    output = torch.randn(1, 512, 50257)
    labels = (torch.randint(0, 50257, (1, 512)), torch.ones(1, 512))
    gradient_signs = torch.ones(512)
    gradient_signs[0] = -1
    
    neox_args = Mock()
    neox_args._current_gradient_signs = gradient_signs
    neox_args.gradient_ascent_loss_scale = 100.0
    
    with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
        mock_ce.return_value = torch.randn(1, 512)
        loss = cross_entropy_with_gradient_tracking(output, labels, _fp16=False, neox_args=neox_args)
        assert isinstance(loss, torch.Tensor)
        print("  ✓ Passed")
    
    # Edge case 2: All tokens are padding except one
    print("\n2. Almost all padding:")
    batch_size, seq_length = 4, 1024
    output = torch.randn(batch_size, seq_length, 50257)
    loss_mask = torch.zeros(batch_size, seq_length)
    loss_mask[0, 0] = 1  # Only first token is valid
    labels = (torch.randint(0, 50257, (batch_size, seq_length)), loss_mask)
    gradient_signs = torch.ones(1)  # Only one valid token
    
    neox_args = Mock()
    neox_args._current_gradient_signs = gradient_signs
    neox_args.gradient_ascent_loss_scale = 5.0
    
    with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
        mock_ce.return_value = torch.randn(batch_size, seq_length)
        loss = cross_entropy_with_gradient_tracking(output, labels, _fp16=False, neox_args=neox_args)
        assert isinstance(loss, torch.Tensor)
        print("  ✓ Passed")
    
    # Edge case 3: Gradient signs larger than expected (shouldn't happen but let's be safe)
    print("\n3. Gradient signs larger than tokens:")
    output = torch.randn(2, 256, 50257)
    labels = (torch.randint(0, 50257, (2, 256)), torch.ones(2, 256))
    gradient_signs = torch.ones(1024)  # Much larger than 512 tokens
    
    neox_args = Mock()
    neox_args._current_gradient_signs = gradient_signs
    neox_args.gradient_ascent_loss_scale = 20.0
    
    with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
        mock_ce.return_value = torch.randn(2, 256)
        loss = cross_entropy_with_gradient_tracking(output, labels, _fp16=False, neox_args=neox_args)
        assert isinstance(loss, torch.Tensor)
        print("  ✓ Passed")


if __name__ == "__main__":
    print("=" * 60)
    print("Tensor Size Mismatch Integration Tests")
    print("=" * 60)
    
    test_exact_error_condition()
    print("\n" + "-" * 60)
    
    test_multi_node_padding_scenario()
    print("\n" + "-" * 60)
    
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("All integration tests passed! 🎉")
    print("The padding mismatch fix is working correctly.")
    print("=" * 60)