#!/usr/bin/env python
"""
Quick test to verify gradient ascent behavior without full model setup.
"""

import torch
import torch.nn.functional as F

def simple_cross_entropy_with_signs(logits, labels, gradient_signs=None):
    """Simplified version of cross entropy with gradient signs"""
    # Standard cross entropy loss
    losses = F.cross_entropy(logits, labels, reduction='none')
    
    # Apply gradient signs if provided
    if gradient_signs is not None:
        losses = losses * gradient_signs
    
    return losses.mean()

# Test setup
torch.manual_seed(42)
batch_size = 4
seq_len = 10
vocab_size = 100

# Create data
logits = torch.randn(batch_size, vocab_size, requires_grad=True)
labels = torch.randint(0, vocab_size, (batch_size,))

# Half samples for ascent (-1), half for descent (1)
gradient_signs = torch.ones(batch_size)
gradient_signs[:2] = -1

print("=== Quick Gradient Ascent Test ===\n")

# Test 1: Check individual losses before optimization
with torch.no_grad():
    individual_losses = F.cross_entropy(logits, labels, reduction='none')
    print("Initial losses:")
    for i in range(batch_size):
        sign_type = "ASCENT" if gradient_signs[i] < 0 else "DESCENT"
        print(f"  Sample {i} ({sign_type}): {individual_losses[i].item():.4f}")

# Test 2: Compute gradients with signs
loss = simple_cross_entropy_with_signs(logits, labels, gradient_signs)
print(f"\nLoss being optimized: {loss.item():.4f}")
loss.backward()

# Test 3: Take an optimization step
learning_rate = 0.1
with torch.no_grad():
    logits_updated = logits - learning_rate * logits.grad
    
    # Check losses after step
    losses_after = F.cross_entropy(logits_updated, labels, reduction='none')
    print(f"\nLosses after gradient step (LR={learning_rate}):")
    for i in range(batch_size):
        sign_type = "ASCENT" if gradient_signs[i] < 0 else "DESCENT"
        change = losses_after[i].item() - individual_losses[i].item()
        print(f"  Sample {i} ({sign_type}): {individual_losses[i].item():.4f} -> {losses_after[i].item():.4f} (change: {change:+.4f})")

print("\n=== Analysis ===")
ascent_changes = [losses_after[i].item() - individual_losses[i].item() for i in range(2)]
descent_changes = [losses_after[i].item() - individual_losses[i].item() for i in range(2, 4)]

print(f"Average change for ASCENT samples: {sum(ascent_changes)/len(ascent_changes):+.4f}")
print(f"Average change for DESCENT samples: {sum(descent_changes)/len(descent_changes):+.4f}")
print("\nExpected: ASCENT should have positive change (loss increases)")
print("Expected: DESCENT should have negative change (loss decreases)")

# Test 4: Verify the math
print("\n=== Gradient Analysis ===")
# Recompute gradients for analysis
logits.grad = None
loss_positive = F.cross_entropy(logits, labels)
loss_positive.backward()
grad_positive = logits.grad.clone()

logits.grad = None  
loss_negative = -F.cross_entropy(logits, labels)
loss_negative.backward()
grad_negative = logits.grad.clone()

print(f"Gradient norm (positive loss): {grad_positive.norm():.4f}")
print(f"Gradient norm (negative loss): {grad_negative.norm():.4f}")
print(f"Gradients are opposite: {torch.allclose(grad_positive, -grad_negative)}")

# The key insight
print("\n=== Key Insight ===")
print("Current implementation: loss = cross_entropy * gradient_signs")
print("When gradient_signs = -1, loss becomes negative")
print("Minimizing negative loss = maximizing positive loss")
print("This SHOULD work in theory, but let's check if it actually does...")