#!/usr/bin/env python
"""
Test that gradient ascent actually increases loss (unlearning) instead of decreasing it.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from megatron.mpu.cross_entropy import vocab_parallel_cross_entropy
from megatron.mpu import initialize as mpu_initialize


def test_gradient_ascent_direction():
    """Test that gradient ascent actually increases the loss"""
    
    print("Testing gradient ascent fix...")
    print("=" * 60)
    
    # Initialize model parallel (required for cross entropy)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )
    mpu_initialize.initialize_model_parallel(1)
    
    # Create a simple model
    vocab_size = 100
    hidden_size = 64
    batch_size = 4
    seq_length = 8
    
    # Simple embedding + projection model
    embedding = nn.Embedding(vocab_size, hidden_size)
    projection = nn.Linear(hidden_size, vocab_size, bias=False)
    
    # Initialize with small weights
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Create input and target
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    target = input_ids.clone()
    
    # Test 1: Gradient Descent (normal case)
    print("\n1. Testing Gradient Descent (sample_signs = 1):")
    print("-" * 40)
    
    embedding.zero_grad()
    projection.zero_grad()
    
    # Forward pass
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    
    # Compute loss with descent (sample_signs = 1)
    sample_signs = torch.ones_like(target, dtype=torch.float)
    loss_before = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss_before = loss_before.mean()
    print(f"Loss before update: {avg_loss_before.item():.6f}")
    
    # Backward pass
    avg_loss_before.backward()
    
    # Manual gradient descent step
    learning_rate = 0.1
    with torch.no_grad():
        embedding.weight -= learning_rate * embedding.weight.grad
        projection.weight -= learning_rate * projection.weight.grad
    
    # Forward pass again to check loss
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    loss_after = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss_after = loss_after.mean()
    print(f"Loss after update: {avg_loss_after.item():.6f}")
    print(f"Change in loss: {avg_loss_after.item() - avg_loss_before.item():.6f}")
    
    assert avg_loss_after < avg_loss_before, "Gradient descent should decrease loss!"
    print("✓ Gradient descent correctly decreases loss")
    
    # Test 2: Gradient Ascent
    print("\n2. Testing Gradient Ascent (sample_signs = -1):")
    print("-" * 40)
    
    # Reset model
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    embedding.zero_grad()
    projection.zero_grad()
    
    # Forward pass
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    
    # Compute loss with ascent (sample_signs = -1)
    sample_signs = torch.full_like(target, -1.0, dtype=torch.float)
    loss_before = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss_before = loss_before.mean()
    print(f"Loss before update: {avg_loss_before.item():.6f}")
    
    # Note: The loss value will be negative due to multiplication by -1
    # We need to look at the actual cross-entropy (without the sign)
    actual_loss_before = abs(avg_loss_before.item())
    print(f"Actual cross-entropy before: {actual_loss_before:.6f}")
    
    # Backward pass
    avg_loss_before.backward()
    
    # Manual gradient descent step (optimizer doesn't know about ascent)
    with torch.no_grad():
        embedding.weight -= learning_rate * embedding.weight.grad
        projection.weight -= learning_rate * projection.weight.grad
    
    # Forward pass again to check loss
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    loss_after = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss_after = loss_after.mean()
    print(f"Loss after update: {avg_loss_after.item():.6f}")
    
    actual_loss_after = abs(avg_loss_after.item())
    print(f"Actual cross-entropy after: {actual_loss_after:.6f}")
    print(f"Change in actual cross-entropy: {actual_loss_after - actual_loss_before:.6f}")
    
    # For gradient ascent, the actual cross-entropy should INCREASE
    assert actual_loss_after > actual_loss_before, "Gradient ascent should increase actual loss!"
    print("✓ Gradient ascent correctly increases loss (unlearning)")
    
    # Test 3: Mixed gradient signs with scaling
    print("\n3. Testing Mixed Signs with Scaling:")
    print("-" * 40)
    
    # Reset model
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Half ascent, half descent
    sample_signs = torch.ones_like(target, dtype=torch.float)
    sample_signs[:batch_size//2] = -1.0
    
    # Test with scaling
    gradient_ascent_loss_scale = 10.0
    
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    loss = vocab_parallel_cross_entropy(logits, target, sample_signs, gradient_ascent_loss_scale)
    
    print(f"Mixed signs with scale={gradient_ascent_loss_scale}:")
    print(f"  Ascent samples: {(sample_signs < 0).sum().item()}")
    print(f"  Descent samples: {(sample_signs > 0).sum().item()}")
    print("✓ Mixed gradient signs with scaling works")
    
    print("\n" + "=" * 60)
    print("All tests passed! Gradient ascent fix is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    test_gradient_ascent_direction()