#!/usr/bin/env python
"""
Test vocab_parallel_cross_entropy directly in a minimal setup.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from megatron.mpu import initialize as mpu_initialize
from megatron.mpu.cross_entropy import vocab_parallel_cross_entropy


def test_vocab_parallel_ce_minimal():
    """Minimal test of vocab_parallel_cross_entropy"""
    
    print("MINIMAL VOCAB PARALLEL CE TEST")
    print("=" * 60)
    
    # Initialize distributed (required)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='tcp://127.0.0.1:29503',
            rank=0,
            world_size=1
        )
    mpu_initialize.initialize_model_parallel(1)
    
    # Create minimal test case
    torch.manual_seed(42)
    batch_size = 1
    seq_length = 1
    vocab_size = 10
    
    # Create a simple linear layer as our "model"
    model = nn.Linear(vocab_size, vocab_size, bias=False)
    nn.init.normal_(model.weight, std=0.1)
    
    # Create input (one-hot encoded)
    input_ids = torch.tensor([[3]])  # Target class 3
    one_hot = torch.zeros(batch_size, seq_length, vocab_size)
    one_hot[0, 0, input_ids[0, 0]] = 1.0
    
    # Forward pass
    logits = model(one_hot)
    target = input_ids
    
    print(f"Input: class {input_ids[0, 0].item()}")
    print(f"Logits: {logits[0, 0].data}")
    print(f"Target: {target[0, 0].item()}")
    
    # Test 1: Normal CE (descent)
    print("\n1. NORMAL CROSS ENTROPY (Descent):")
    print("-" * 40)
    
    # Compute initial CE manually
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        ce_initial = -torch.log(probs[0, 0, target[0, 0]])
        print(f"Initial CE: {ce_initial.item():.6f}")
    
    # Forward with descent
    model.zero_grad()
    sample_signs = torch.ones_like(target, dtype=torch.float)
    loss = vocab_parallel_cross_entropy(logits, target, sample_signs)
    print(f"Loss returned: {loss[0, 0].item():.6f}")
    
    # Backward
    loss.mean().backward()
    
    # Update
    with torch.no_grad():
        model.weight -= 0.1 * model.weight.grad
    
    # Check new CE
    with torch.no_grad():
        logits_new = model(one_hot)
        probs_new = torch.softmax(logits_new, dim=-1)
        ce_final = -torch.log(probs_new[0, 0, target[0, 0]])
        print(f"Final CE: {ce_final.item():.6f}")
        print(f"Change: {ce_final.item() - ce_initial.item():.6f} (should be negative)")
        
    descent_works = ce_final < ce_initial
    print(f"Descent works: {descent_works}")
    
    # Test 2: Gradient Ascent
    print("\n2. GRADIENT ASCENT:")
    print("-" * 40)
    
    # Reset model
    torch.manual_seed(42)
    model = nn.Linear(vocab_size, vocab_size, bias=False)
    nn.init.normal_(model.weight, std=0.1)
    
    # Compute initial CE
    with torch.no_grad():
        logits = model(one_hot)
        probs = torch.softmax(logits, dim=-1)
        ce_initial = -torch.log(probs[0, 0, target[0, 0]])
        print(f"Initial CE: {ce_initial.item():.6f}")
    
    # Forward with ascent
    model.zero_grad()
    logits = model(one_hot)
    sample_signs = torch.full_like(target, -1.0, dtype=torch.float)
    loss = vocab_parallel_cross_entropy(logits, target, sample_signs)
    print(f"Loss returned: {loss[0, 0].item():.6f} (negative)")
    
    # Backward
    loss.mean().backward()
    
    # Update (standard optimizer step)
    with torch.no_grad():
        model.weight -= 0.1 * model.weight.grad
    
    # Check new CE
    with torch.no_grad():
        logits_new = model(one_hot)
        probs_new = torch.softmax(logits_new, dim=-1)
        ce_final = -torch.log(probs_new[0, 0, target[0, 0]])
        print(f"Final CE: {ce_final.item():.6f}")
        print(f"Change: {ce_final.item() - ce_initial.item():.6f} (should be positive)")
        
    ascent_works = ce_final > ce_initial
    print(f"Ascent works: {ascent_works}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"- Gradient descent: {'✓ WORKS' if descent_works else '✗ FAILED'}")
    print(f"- Gradient ascent: {'✓ WORKS' if ascent_works else '✗ FAILED'}")
    
    if descent_works and ascent_works:
        print("\n✓ GRADIENT ASCENT IS WORKING CORRECTLY!")
        print("The implementation is correct.")
    else:
        print("\n✗ THERE IS STILL AN ISSUE!")
        
    # Additional debugging
    print("\n" + "=" * 60)
    print("DEBUGGING INFO:")
    
    # Check gradient signs
    print(f"\nFor reference, model.weight.grad stats:")
    print(f"- Shape: {model.weight.grad.shape}")
    print(f"- Mean: {model.weight.grad.mean().item():.6f}")
    print(f"- Std: {model.weight.grad.std().item():.6f}")
    

if __name__ == "__main__":
    test_vocab_parallel_ce_minimal()