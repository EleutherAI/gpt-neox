#!/usr/bin/env python
"""
Minimal test to verify gradient ascent implementation.
This directly tests the cross_entropy function with gradient signs.
"""

import torch
import sys
sys.path.append('/workspace/local_repos/gpt-neox')

from megatron.mpu import vocab_parallel_cross_entropy
from megatron.model.gpt2_model import cross_entropy

# Set up minimal test
torch.manual_seed(42)
vocab_size = 100
batch_size = 4
seq_len = 10

# Create fake outputs and labels
outputs = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
labels = torch.randint(0, vocab_size, (batch_size, seq_len))
loss_mask = torch.ones(batch_size, seq_len)

# Create gradient signs: half ascent (-1), half descent (1)
gradient_signs = torch.ones(batch_size, seq_len)
gradient_signs[:2, :] = -1  # First 2 samples are ascent

print("=== Testing Gradient Ascent Implementation ===")

# Test 1: Forward pass with gradient signs
loss_with_signs = cross_entropy(outputs, (labels, loss_mask), _fp16=False, sample_signs=gradient_signs)
print(f"\nLoss with gradient signs: {loss_with_signs.item():.4f}")

# Test 2: Check individual losses
with torch.no_grad():
    # Calculate per-sample losses
    sample_losses = []
    for i in range(batch_size):
        sample_output = outputs[i:i+1]
        sample_label = labels[i:i+1]
        sample_mask = loss_mask[i:i+1]
        sample_loss = cross_entropy(sample_output, (sample_label, sample_mask), _fp16=False)
        sample_losses.append(sample_loss.item())
    
    print(f"\nPer-sample losses (without signs):")
    for i, loss in enumerate(sample_losses):
        sign = "ASCENT" if gradient_signs[i, 0] < 0 else "DESCENT"
        print(f"  Sample {i} ({sign}): {loss:.4f}")

# Test 3: Backward pass and check gradients
outputs.grad = None
loss_with_signs.backward()

# Check gradient magnitudes for ascent vs descent samples
grad_norm_ascent = outputs.grad[:2].norm().item()
grad_norm_descent = outputs.grad[2:].norm().item()

print(f"\nGradient norms:")
print(f"  Ascent samples (first 2): {grad_norm_ascent:.4f}")
print(f"  Descent samples (last 2): {grad_norm_descent:.4f}")

# Test 4: Verify gradient directions
# Take a small optimization step and see if losses change as expected
with torch.no_grad():
    outputs_updated = outputs - 0.1 * outputs.grad  # Gradient descent step
    
    print(f"\nLosses after one gradient step:")
    for i in range(batch_size):
        sample_output = outputs_updated[i:i+1]
        sample_label = labels[i:i+1]
        sample_mask = loss_mask[i:i+1]
        sample_loss = cross_entropy(sample_output, (sample_label, sample_mask), _fp16=False)
        original_loss = sample_losses[i]
        change = sample_loss.item() - original_loss
        sign = "ASCENT" if gradient_signs[i, 0] < 0 else "DESCENT"
        print(f"  Sample {i} ({sign}): {original_loss:.4f} -> {sample_loss.item():.4f} (change: {change:+.4f})")
    
    print(f"\nExpected behavior:")
    print(f"  - ASCENT samples: loss should INCREASE (positive change)")
    print(f"  - DESCENT samples: loss should DECREASE (negative change)")

# Test 5: Check the actual loss value being optimized
print(f"\n=== Debugging Info ===")
print(f"Loss value being backpropagated: {loss_with_signs.item():.4f}")
if loss_with_signs.item() < 0:
    print("WARNING: Loss is negative! This might cause optimizer issues.")

# Test per-token losses with signs applied
with torch.no_grad():
    per_token_losses = vocab_parallel_cross_entropy(outputs, labels)
    losses_with_signs = per_token_losses * gradient_signs
    print(f"\nPer-token losses shape: {per_token_losses.shape}")
    print(f"First few losses (original): {per_token_losses[0, :5].tolist()}")
    print(f"First few losses (with signs): {losses_with_signs[0, :5].tolist()}")
    print(f"Mean loss for ascent tokens: {losses_with_signs[gradient_signs < 0].mean().item():.4f}")
    print(f"Mean loss for descent tokens: {losses_with_signs[gradient_signs > 0].mean().item():.4f}")