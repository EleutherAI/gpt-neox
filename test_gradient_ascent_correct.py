#!/usr/bin/env python
"""
Test gradient ascent implementation after fixing the double negation bug.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from megatron.mpu.cross_entropy import vocab_parallel_cross_entropy
from megatron.mpu import initialize as mpu_initialize


def test_gradient_ascent_fixed():
    """Test that gradient ascent works correctly after the fix"""
    
    print("Testing FIXED gradient ascent implementation...")
    print("=" * 60)
    
    # Initialize model parallel
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
    torch.manual_seed(42)
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Create input and target
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    target = input_ids.clone()
    
    # Helper function to compute actual cross entropy
    def compute_actual_cross_entropy():
        with torch.no_grad():
            embeddings = embedding(input_ids)
            logits = projection(embeddings)
            # Compute cross entropy without any signs
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            loss = ce_loss(logits.view(-1, vocab_size), target.view(-1))
            return loss.view(batch_size, seq_length).mean().item()
    
    print("\n1. Testing Gradient DESCENT (baseline):")
    print("-" * 40)
    
    # Reset model
    torch.manual_seed(42)
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Initial cross entropy
    initial_ce = compute_actual_cross_entropy()
    print(f"Initial cross-entropy: {initial_ce:.6f}")
    
    # Forward pass with descent
    embedding.zero_grad()
    projection.zero_grad()
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    
    # Gradient descent (sample_signs = 1)
    sample_signs = torch.ones_like(target, dtype=torch.float)
    loss = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss = loss.mean()
    avg_loss.backward()
    
    # Update
    learning_rate = 0.1
    with torch.no_grad():
        embedding.weight -= learning_rate * embedding.weight.grad
        projection.weight -= learning_rate * projection.weight.grad
    
    # Check new cross entropy
    final_ce = compute_actual_cross_entropy()
    print(f"Final cross-entropy: {final_ce:.6f}")
    print(f"Change: {final_ce - initial_ce:.6f} (should be negative)")
    assert final_ce < initial_ce, "Gradient descent should decrease cross-entropy!"
    print("✓ Gradient descent works correctly")
    
    print("\n2. Testing Gradient ASCENT (with fix):")
    print("-" * 40)
    
    # Reset model
    torch.manual_seed(42)
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Initial cross entropy
    initial_ce = compute_actual_cross_entropy()
    print(f"Initial cross-entropy: {initial_ce:.6f}")
    
    # Forward pass with ascent
    embedding.zero_grad()
    projection.zero_grad()
    embeddings = embedding(input_ids)
    logits = projection(embeddings)
    
    # Gradient ascent (sample_signs = -1)
    sample_signs = torch.full_like(target, -1.0, dtype=torch.float)
    loss = vocab_parallel_cross_entropy(logits, target, sample_signs)
    avg_loss = loss.mean()
    print(f"Modified loss for optimization: {avg_loss.item():.6f} (negative)")
    
    # Backward pass
    avg_loss.backward()
    
    # Standard optimizer update (it doesn't know about ascent)
    with torch.no_grad():
        embedding.weight -= learning_rate * embedding.weight.grad
        projection.weight -= learning_rate * projection.weight.grad
    
    # Check new cross entropy
    final_ce = compute_actual_cross_entropy()
    print(f"Final cross-entropy: {final_ce:.6f}")
    print(f"Change: {final_ce - initial_ce:.6f} (should be positive)")
    assert final_ce > initial_ce, "Gradient ascent should increase cross-entropy!"
    print("✓ Gradient ascent works correctly")
    
    print("\n3. Testing with loss scaling:")
    print("-" * 40)
    
    # Reset model
    torch.manual_seed(42)
    nn.init.normal_(embedding.weight, std=0.02)
    nn.init.normal_(projection.weight, std=0.02)
    
    # Test with different scales
    for scale in [1.0, 5.0, 10.0]:
        embedding.zero_grad()
        projection.zero_grad()
        
        initial_ce = compute_actual_cross_entropy()
        
        embeddings = embedding(input_ids)
        logits = projection(embeddings)
        
        # Half ascent, half descent
        sample_signs = torch.ones_like(target, dtype=torch.float)
        sample_signs[:batch_size//2] = -1.0
        
        loss = vocab_parallel_cross_entropy(logits, target, sample_signs, scale)
        loss.mean().backward()
        
        # Small update to see the effect
        with torch.no_grad():
            embedding.weight -= 0.01 * embedding.weight.grad
            projection.weight -= 0.01 * projection.weight.grad
        
        final_ce = compute_actual_cross_entropy()
        print(f"Scale={scale}: CE change = {final_ce - initial_ce:.6f}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: Gradient ascent is now working correctly!")
    print("The fix removed the double negation bug.")
    print("=" * 60)


if __name__ == "__main__":
    test_gradient_ascent_fixed()