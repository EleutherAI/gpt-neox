#!/usr/bin/env python
"""
Comprehensive tests for gradient ascent loss scaling functionality.
Tests both the mathematical correctness and integration with GPT-NeoX training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.mpu import vocab_parallel_cross_entropy
from megatron.model.gpt2_model import cross_entropy
from megatron.mpu.initialize import (
    initialize_model_parallel, 
    get_model_parallel_group,
    model_parallel_is_initialized,
    destroy_model_parallel
)


class TestGradientAscentLossScale:
    """Test suite for gradient ascent loss scaling"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test"""
        # Setup
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='gloo',
                init_method='tcp://127.0.0.1:29501',
                rank=0,
                world_size=1
            )
        
        if not model_parallel_is_initialized():
            initialize_model_parallel(1)
        
        yield
        
        # Cleanup
        destroy_model_parallel()
    
    def test_basic_loss_scaling(self):
        """Test that gradient ascent loss scaling works correctly"""
        torch.manual_seed(42)
        
        # Create test data
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create gradient signs: half ascent, half descent
        gradient_signs = torch.ones(batch_size, seq_len)
        gradient_signs[:2, :] = -1  # First 2 samples are ascent
        
        # Test with different scaling factors
        scaling_factors = [1.0, 5.0, 10.0, 20.0]
        losses = []
        
        for scale in scaling_factors:
            loss = vocab_parallel_cross_entropy(
                logits, labels, gradient_signs, gradient_ascent_loss_scale=scale
            )
            losses.append(loss.mean().item())
        
        # Verify that losses scale appropriately
        # With 50% ascent samples, the effect should be noticeable
        print("\n=== Loss Scaling Test ===")
        for i, (scale, loss) in enumerate(zip(scaling_factors, losses)):
            print(f"Scale: {scale:4.1f}, Loss: {loss:8.4f}")
            
        # The loss should change as we increase the scaling factor
        assert losses[1] != losses[0], "Loss should change with scaling"
        assert abs(losses[2]) > abs(losses[1]), "Higher scale should have larger effect"
    
    def test_gradient_direction_with_scaling(self):
        """Test that gradients are correctly scaled for ascent samples"""
        torch.manual_seed(42)
        
        # Simple model
        model = nn.Linear(10, 50, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # Create data
        batch_size = 4
        x = torch.randn(batch_size, 10)
        labels = torch.randint(0, 50, (batch_size,))
        
        # Half ascent, half descent
        gradient_signs = torch.ones(batch_size)
        gradient_signs[:2] = -1
        
        # Test different scales
        scales = [1.0, 10.0]
        param_changes = []
        
        for scale in scales:
            # Reset model
            model.weight.data = torch.randn_like(model.weight)
            initial_weight = model.weight.data.clone()
            
            # Forward pass
            output = model(x)
            
            # Compute loss with scaling
            losses = F.cross_entropy(output, labels, reduction='none')
            if scale != 1.0:
                scaled_signs = gradient_signs.clone()
                scaled_signs[gradient_signs < 0] *= scale
                losses = losses * scaled_signs
            else:
                losses = losses * gradient_signs
            loss = losses.mean()
            
            # Backward and step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Measure parameter change
            param_change = (model.weight.data - initial_weight).norm().item()
            param_changes.append(param_change)
        
        print("\n=== Gradient Scaling Test ===")
        print(f"Parameter change with scale=1.0: {param_changes[0]:.4f}")
        print(f"Parameter change with scale=10.0: {param_changes[1]:.4f}")
        print(f"Ratio: {param_changes[1]/param_changes[0]:.2f}x")
        
        # With higher scaling, we should see larger parameter changes
        assert param_changes[1] > param_changes[0] * 2, "Scaling should amplify gradients"
    
    def test_loss_mask_interaction(self):
        """Test that loss scaling works correctly with loss masks"""
        torch.manual_seed(42)
        
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        # Create data
        outputs = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Create loss mask (some tokens are masked)
        loss_mask = torch.ones(batch_size, seq_len)
        loss_mask[:, 5:] = 0  # Mask out second half of sequences
        
        # Gradient signs
        gradient_signs = torch.ones(batch_size, seq_len)
        gradient_signs[0, :] = -1  # First sample is all ascent
        gradient_signs[2, :] = -1  # Third sample is all ascent
        
        # Test with scaling
        scale = 10.0
        loss = cross_entropy(
            outputs, (labels, loss_mask), 
            _fp16=False, 
            sample_signs=gradient_signs,
            gradient_ascent_loss_scale=scale
        )
        
        print(f"\n=== Loss Mask Test ===")
        print(f"Loss with mask and scaling: {loss.item():.4f}")
        
        # Verify loss is finite and reasonable
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() != 0, "Loss should be non-zero"
    
    def test_mixed_precision_compatibility(self):
        """Test that scaling works with fp16"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.manual_seed(42)
        
        batch_size = 2
        seq_len = 8
        vocab_size = 50
        
        # Create fp16 data
        outputs = torch.randn(batch_size, seq_len, vocab_size, 
                            dtype=torch.float16, device='cuda')
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
        loss_mask = torch.ones(batch_size, seq_len, dtype=torch.float16, device='cuda')
        
        # Gradient signs
        gradient_signs = torch.ones(batch_size, seq_len, device='cuda')
        gradient_signs[0, :] = -1
        
        # Test with scaling
        scale = 5.0
        loss = cross_entropy(
            outputs, (labels, loss_mask),
            _fp16=True,
            sample_signs=gradient_signs,
            gradient_ascent_loss_scale=scale
        )
        
        print(f"\n=== FP16 Test ===")
        print(f"FP16 loss with scaling: {loss.item():.4f}")
        
        assert torch.isfinite(loss), "FP16 loss should be finite"
    
    def test_no_gradient_signs(self):
        """Test that loss computation works without gradient signs"""
        torch.manual_seed(42)
        
        batch_size = 2
        seq_len = 8
        vocab_size = 50
        
        outputs = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        
        # Test without gradient signs (should work normally)
        loss = cross_entropy(
            outputs, (labels, loss_mask),
            _fp16=False,
            sample_signs=None,
            gradient_ascent_loss_scale=10.0  # Should have no effect
        )
        
        print(f"\n=== No Gradient Signs Test ===")
        print(f"Loss without gradient signs: {loss.item():.4f}")
        
        assert torch.isfinite(loss), "Loss should be finite"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_extreme_scaling_factors(self):
        """Test behavior with extreme scaling factors"""
        torch.manual_seed(42)
        
        batch_size = 2
        seq_len = 8
        vocab_size = 50
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        gradient_signs = torch.ones(batch_size, seq_len)
        gradient_signs[0, :] = -1
        
        # Test extreme scales
        extreme_scales = [0.1, 100.0, 1000.0]
        
        print("\n=== Extreme Scaling Test ===")
        for scale in extreme_scales:
            loss = vocab_parallel_cross_entropy(
                logits, labels, gradient_signs, gradient_ascent_loss_scale=scale
            )
            print(f"Scale: {scale:7.1f}, Loss mean: {loss.mean().item():10.4f}")
            
            # Even with extreme scales, loss should be finite
            assert torch.isfinite(loss).all(), f"Loss should be finite for scale={scale}"
    
    def test_backward_pass_with_scaling(self):
        """Test that backward pass works correctly with scaling"""
        torch.manual_seed(42)
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self, vocab_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 128)
                self.output = nn.Linear(128, vocab_size)
            
            def forward(self, x):
                return self.output(self.embedding(x))
        
        model = SimpleModel(100)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create data
        batch_size = 4
        seq_len = 10
        inputs = torch.randint(0, 100, (batch_size, seq_len))
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Gradient signs - alternating samples
        gradient_signs = torch.ones(batch_size, seq_len)
        gradient_signs[::2, :] = -1  # Every other sample is ascent
        
        # Track losses over iterations
        losses_no_scale = []
        losses_with_scale = []
        
        # Train without scaling
        for _ in range(5):
            outputs = model(inputs)
            losses = F.cross_entropy(outputs.view(-1, 100), labels.view(-1), reduction='none')
            losses = losses.view(batch_size, seq_len)
            loss = (losses * gradient_signs).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track raw losses
            with torch.no_grad():
                raw_loss = losses.mean().item()
                losses_no_scale.append(raw_loss)
        
        # Reset model
        model = SimpleModel(100)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train with scaling
        scale = 10.0
        for _ in range(5):
            outputs = model(inputs)
            losses = F.cross_entropy(outputs.view(-1, 100), labels.view(-1), reduction='none')
            losses = losses.view(batch_size, seq_len)
            
            # Apply scaling to ascent samples
            scaled_signs = gradient_signs.clone()
            scaled_signs[gradient_signs < 0] *= scale
            loss = (losses * scaled_signs).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track raw losses
            with torch.no_grad():
                raw_loss = losses.mean().item()
                losses_with_scale.append(raw_loss)
        
        print("\n=== Training with Scaling Test ===")
        print("Losses without scaling:", [f"{l:.4f}" for l in losses_no_scale])
        print("Losses with scaling:   ", [f"{l:.4f}" for l in losses_with_scale])
        
        # With scaling on half the samples, we should see different training dynamics
        assert losses_with_scale[-1] != losses_no_scale[-1], "Scaling should affect training"


def test_integration_with_neox_args():
    """Test that gradient_ascent_loss_scale integrates with NeoXArgs"""
    from megatron.neox_arguments import NeoXArgs
    
    # Create a minimal config
    config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 8,
        "seq_length": 512,
        "max_position_embeddings": 512,
        "vocab_size": 50304,
        "tokenizer_type": "GPT2BPETokenizer",
        "gradient_ascent_loss_scale": 15.0,  # Our new parameter
    }
    
    # Create NeoXArgs instance
    neox_args = NeoXArgs.from_dict(config)
    
    # Verify the parameter is set correctly
    assert hasattr(neox_args, 'gradient_ascent_loss_scale'), "NeoXArgs should have gradient_ascent_loss_scale"
    assert neox_args.gradient_ascent_loss_scale == 15.0, "Value should be set correctly"
    
    print("\n=== NeoXArgs Integration Test ===")
    print(f"gradient_ascent_loss_scale: {neox_args.gradient_ascent_loss_scale}")
    print("Integration test passed!")


if __name__ == "__main__":
    # Run tests
    print("Running Gradient Ascent Loss Scale Tests...\n")
    
    # Run integration test first (doesn't need pytest)
    test_integration_with_neox_args()
    
    # Run pytest tests
    pytest.main([__file__, "-v", "-s"])