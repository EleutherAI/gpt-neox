#!/usr/bin/env python
"""
Trace the exact gradient flow to understand what's happening.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def trace_gradient_flow():
    """Trace gradient flow step by step"""
    
    print("TRACING GRADIENT FLOW")
    print("=" * 60)
    
    # Simple test case
    torch.manual_seed(42)
    
    # Single parameter model
    w = nn.Parameter(torch.tensor([[2.0]]))
    
    # Simple loss: L = w^2
    def compute_loss():
        return (w ** 2).sum()
    
    print("Initial state:")
    print(f"w = {w.item():.4f}")
    print(f"L(w) = w² = {compute_loss().item():.4f}")
    
    # Test 1: Standard gradient descent
    print("\n1. STANDARD GRADIENT DESCENT:")
    print("-" * 40)
    
    loss = compute_loss()
    print(f"Loss = {loss.item():.4f}")
    
    w.grad = None
    loss.backward()
    print(f"Gradient dL/dw = {w.grad.item():.4f}")
    
    # Update
    with torch.no_grad():
        w_new = w - 0.5 * w.grad
        print(f"Update: w_new = w - 0.5 * grad = {w.item():.4f} - 0.5 * {w.grad.item():.4f} = {w_new.item():.4f}")
        w.copy_(w_new)
    
    print(f"New loss = {compute_loss().item():.4f} (decreased ✓)")
    
    # Reset
    w.data = torch.tensor([[2.0]])
    
    # Test 2: Our gradient ascent approach
    print("\n2. OUR GRADIENT ASCENT (multiply loss by -1):")
    print("-" * 40)
    
    loss = compute_loss()
    modified_loss = -1.0 * loss  # This is what our forward pass does
    print(f"Original loss = {loss.item():.4f}")
    print(f"Modified loss = {modified_loss.item():.4f}")
    
    w.grad = None
    modified_loss.backward()
    print(f"Gradient d(modified_loss)/dw = {w.grad.item():.4f}")
    
    # Standard optimizer update (doesn't know about the sign)
    with torch.no_grad():
        w_new = w - 0.5 * w.grad
        print(f"Update: w_new = w - 0.5 * grad = {w.item():.4f} - 0.5 * {w.grad.item():.4f} = {w_new.item():.4f}")
        w.copy_(w_new)
    
    print(f"New loss = {compute_loss().item():.4f} (should increase)")
    
    # Now let's trace what happens in our actual cross entropy
    print("\n3. ACTUAL CROSS ENTROPY IMPLEMENTATION:")
    print("-" * 40)
    
    # Import the actual function
    from megatron.mpu.cross_entropy import _VocabParallelCrossEntropy
    
    # Check if the backward pass is modifying gradients
    print("\nChecking backward pass implementation...")
    
    # Read the source to see what's actually there
    import inspect
    backward_source = inspect.getsource(_VocabParallelCrossEntropy.backward)
    
    print("Backward pass source code:")
    print("-" * 30)
    for i, line in enumerate(backward_source.split('\n')[0:20]):  # First 20 lines
        print(f"{i+1:3d}: {line}")
    
    # Check if sample_signs is used in backward
    if "sample_signs" in backward_source:
        print("\n⚠️  WARNING: backward pass references sample_signs")
        if "grad_input.mul_" in backward_source and "sample_signs" in backward_source:
            print("⚠️  ERROR: backward pass is multiplying by sample_signs!")
    else:
        print("\n✓ Good: backward pass does not use sample_signs")


def manual_gradient_check():
    """Manually check gradient computation"""
    print("\n\nMANUAL GRADIENT CHECK")
    print("=" * 60)
    
    # Simulate cross entropy gradient
    vocab_size = 10
    batch_size = 1
    
    # Create logits and target
    logits = torch.randn(batch_size, vocab_size, requires_grad=True)
    target = torch.tensor([3])  # Target class 3
    
    # Manual cross entropy
    softmax = torch.nn.functional.softmax(logits, dim=-1)
    ce_loss = -torch.log(softmax[0, target])
    
    print(f"Logits: {logits.data}")
    print(f"Target: {target.item()}")
    print(f"Softmax: {softmax.data}")
    print(f"CE Loss: {ce_loss.item():.4f}")
    
    # Gradient with respect to logits
    ce_loss.backward()
    print(f"\nGradient (descent):")
    print(f"dL/d(logits) = {logits.grad}")
    
    # For ascent, we want -gradient
    print(f"\nFor gradient ascent, we need:")
    print(f"-dL/d(logits) = {-logits.grad}")
    
    # What happens if we multiply loss by -1?
    logits.grad = None
    modified_loss = -ce_loss
    modified_loss.backward()
    print(f"\nWith modified loss = -CE:")
    print(f"d(-L)/d(logits) = {logits.grad}")
    print("This should equal -dL/d(logits) from above ✓")


if __name__ == "__main__":
    trace_gradient_flow()
    manual_gradient_check()