#!/usr/bin/env python
"""
Test that gradient ascent loss scaling handles padding mismatches correctly.
"""

import torch
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from megatron.model.gpt2_model import cross_entropy_with_gradient_tracking


class TestGradientAscentPaddingFix:
    """Test suite for gradient ascent with padding mismatches"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.seq_length = 2048
        self.vocab_size = 50257
        
    def test_mismatched_tensor_sizes(self):
        """Test that the function handles mismatched tensor sizes correctly"""
        # Create tensors with different sizes (simulating padding mismatch)
        actual_tokens = self.batch_size * self.seq_length - 32  # 32 padding tokens
        
        # Mock output logits
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        
        # Labels and loss mask (full size)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length)
        labels = (labels_tensor, loss_mask)
        
        # Gradient signs (smaller size - no padding)
        gradient_signs = torch.ones(actual_tokens)
        # Make some tokens ascent
        gradient_signs[:100] = -1
        
        # Mock neox_args
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 10.0
        
        # Mock the cross_entropy function
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            # Return losses of the correct size
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length)
            
            # This should not raise an error despite size mismatch
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            # Verify the function was called
            assert mock_ce.called
            assert isinstance(loss, torch.Tensor)
            
            # Check that additional losses were computed
            assert hasattr(neox_args, '_additional_losses')
            assert 'gradient_ascent_loss' in neox_args._additional_losses
            assert 'gradient_descent_loss' in neox_args._additional_losses
    
    def test_exact_match_tensor_sizes(self):
        """Test that the function works correctly when tensor sizes match exactly"""
        # Create tensors with matching sizes
        total_tokens = self.batch_size * self.seq_length
        
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length)
        labels = (labels_tensor, loss_mask)
        
        # Gradient signs (same size)
        gradient_signs = torch.ones(total_tokens)
        gradient_signs[:500] = -1  # Some ascent tokens
        
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 5.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length)
            
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            assert hasattr(neox_args, '_additional_losses')
    
    def test_extreme_padding_mismatch(self):
        """Test with extreme padding where gradient signs are much smaller"""
        # Simulate case where half the tokens are padding
        actual_tokens = (self.batch_size * self.seq_length) // 2
        
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length)
        # Set second half as padding
        loss_mask.view(-1)[actual_tokens:] = 0
        labels = (labels_tensor, loss_mask)
        
        gradient_signs = torch.ones(actual_tokens)
        gradient_signs[:50] = -1
        
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 20.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length)
            
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            # Verify metrics only computed on valid tokens
            assert neox_args._additional_losses['gradient_ascent_samples'] == 50
    
    def test_no_gradient_signs(self):
        """Test that function works without gradient signs (normal mode)"""
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length)
        labels = (labels_tensor, loss_mask)
        
        # No gradient signs
        neox_args = Mock()
        neox_args.gradient_ascent_loss_scale = 10.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length)
            
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            # No additional losses should be computed
            assert not hasattr(neox_args, '_additional_losses')
    
    def test_fp16_mode_with_padding(self):
        """Test FP16 mode with padding mismatch"""
        actual_tokens = self.batch_size * self.seq_length - 64
        
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size, dtype=torch.half)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.half)
        labels = (labels_tensor, loss_mask)
        
        gradient_signs = torch.ones(actual_tokens)
        gradient_signs[::10] = -1  # Every 10th token is ascent
        
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 15.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length, dtype=torch.half)
            
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=True, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            assert loss.dtype == torch.half
    
    def test_all_ascent_tokens_with_padding(self):
        """Test when all non-padding tokens are marked for ascent"""
        actual_tokens = self.batch_size * self.seq_length - 128
        
        output = torch.randn(self.batch_size, self.seq_length, self.vocab_size)
        labels_tensor = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        loss_mask = torch.ones(self.batch_size, self.seq_length)
        labels = (labels_tensor, loss_mask)
        
        # All tokens are ascent
        gradient_signs = torch.full((actual_tokens,), -1.0)
        
        neox_args = Mock()
        neox_args._current_gradient_signs = gradient_signs
        neox_args.gradient_ascent_loss_scale = 10.0
        
        with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
            mock_ce.return_value = torch.randn(self.batch_size, self.seq_length)
            
            loss = cross_entropy_with_gradient_tracking(
                output, labels, _fp16=False, neox_args=neox_args
            )
            
            assert isinstance(loss, torch.Tensor)
            assert 'gradient_ascent_loss' in neox_args._additional_losses
            # No descent loss should be computed
            assert 'gradient_descent_loss' not in neox_args._additional_losses


@pytest.mark.parametrize("padding_tokens,scale", [
    (0, 1.0),      # No padding, no scaling
    (32, 10.0),    # Small padding, moderate scaling  
    (256, 50.0),   # Large padding, high scaling
    (1024, 100.0), # Very large padding, very high scaling
])
def test_various_padding_scenarios(padding_tokens, scale):
    """Test various padding and scaling combinations"""
    batch_size = 8
    seq_length = 2048
    vocab_size = 50257
    actual_tokens = batch_size * seq_length - padding_tokens
    
    output = torch.randn(batch_size, seq_length, vocab_size)
    labels_tensor = torch.randint(0, vocab_size, (batch_size, seq_length))
    loss_mask = torch.ones(batch_size, seq_length)
    if padding_tokens > 0:
        loss_mask.view(-1)[-padding_tokens:] = 0
    labels = (labels_tensor, loss_mask)
    
    gradient_signs = torch.ones(actual_tokens)
    # 2% ascent tokens
    n_ascent = max(1, int(actual_tokens * 0.02))
    gradient_signs[:n_ascent] = -1
    
    neox_args = Mock()
    neox_args._current_gradient_signs = gradient_signs
    neox_args.gradient_ascent_loss_scale = scale
    
    with patch('megatron.mpu.vocab_parallel_cross_entropy') as mock_ce:
        mock_ce.return_value = torch.randn(batch_size, seq_length)
        
        # Should not raise any errors
        loss = cross_entropy_with_gradient_tracking(
            output, labels, _fp16=False, neox_args=neox_args
        )
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert torch.isfinite(loss)


if __name__ == "__main__":
    # Run the tests
    test = TestGradientAscentPaddingFix()
    test.setup_method()
    
    print("Testing mismatched tensor sizes...")
    test.test_mismatched_tensor_sizes()
    print("✓ Passed")
    
    print("Testing exact match tensor sizes...")
    test.test_exact_match_tensor_sizes()
    print("✓ Passed")
    
    print("Testing extreme padding mismatch...")
    test.test_extreme_padding_mismatch()
    print("✓ Passed")
    
    print("Testing no gradient signs...")
    test.test_no_gradient_signs()
    print("✓ Passed")
    
    print("Testing FP16 mode with padding...")
    test.test_fp16_mode_with_padding()
    print("✓ Passed")
    
    print("Testing all ascent tokens with padding...")
    test.test_all_ascent_tokens_with_padding()
    print("✓ Passed")
    
    print("\nTesting various padding scenarios...")
    for padding, scale in [(0, 1.0), (32, 10.0), (256, 50.0), (1024, 100.0)]:
        test_various_padding_scenarios(padding, scale)
        print(f"✓ Padding={padding}, scale={scale}")
    
    print("\nAll tests passed! ✨")