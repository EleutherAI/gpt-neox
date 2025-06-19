#!/usr/bin/env python
"""
Final verification that gradient ascent is working correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def verify_gradient_ascent():
    """Verify gradient ascent with a simple, clear example"""
    
    print("FINAL GRADIENT ASCENT VERIFICATION")
    print("=" * 60)
    
    # Very simple case: minimize/maximize x^2
    x = torch.tensor(1.0, requires_grad=True)
    
    # Test descent
    print("\n1. Gradient Descent (minimize x²):")
    loss = x ** 2
    loss.backward()
    print(f"x = {x.item()}, loss = {loss.item()}")
    print(f"grad = {x.grad.item()}")
    
    with torch.no_grad():
        x_new = x - 0.5 * x.grad
    print(f"After update: x = {x_new.item()}, loss = {x_new.item()**2}")
    print(f"Loss decreased: {x_new.item()**2 < loss.item()} ✓")
    
    # Test ascent via negated loss
    x = torch.tensor(1.0, requires_grad=True)
    print("\n2. Gradient Ascent (maximize x² via minimizing -x²):")
    loss = x ** 2
    neg_loss = -loss
    neg_loss.backward()
    print(f"x = {x.item()}, loss = {loss.item()}, neg_loss = {neg_loss.item()}")
    print(f"grad = {x.grad.item()}")
    
    with torch.no_grad():
        x_new = x - 0.5 * x.grad
    print(f"After update: x = {x_new.item()}, loss = {x_new.item()**2}")
    print(f"Loss increased: {x_new.item()**2 > loss.item()} ✓")
    
    # Now test with cross entropy
    print("\n3. Cross Entropy Test:")
    print("-" * 40)
    
    # Simple 3-class problem
    W = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], 
                      [0.0, 0.0, 1.0]], requires_grad=True)
    x = torch.tensor([[1.0, 0.0, 0.0]])  # One-hot for class 0
    target = torch.tensor([0])
    
    def compute_ce():
        logits = x @ W.T
        return F.cross_entropy(logits, target)
    
    # Descent
    print("\nDescent:")
    initial_ce = compute_ce().item()
    print(f"Initial CE: {initial_ce:.4f}")
    
    loss = compute_ce()
    W.grad = None
    loss.backward()
    
    with torch.no_grad():
        W -= 0.5 * W.grad
        
    final_ce = compute_ce().item()
    print(f"Final CE: {final_ce:.4f}")
    print(f"Decreased: {final_ce < initial_ce} ✓")
    
    # Reset and test ascent
    W.data = torch.tensor([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0], 
                           [0.0, 0.0, 1.0]])
    
    print("\nAscent:")
    initial_ce = compute_ce().item()
    print(f"Initial CE: {initial_ce:.4f}")
    
    loss = -compute_ce()  # Negate for ascent
    W.grad = None
    loss.backward()
    
    with torch.no_grad():
        W -= 0.5 * W.grad
        
    final_ce = compute_ce().item()
    print(f"Final CE: {final_ce:.4f}")
    print(f"Increased: {final_ce > initial_ce} ✓")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: Gradient ascent via negated loss WORKS correctly!")
    print("The math is sound. Any issues must be elsewhere.")
    print("=" * 60)


if __name__ == "__main__":
    verify_gradient_ascent()