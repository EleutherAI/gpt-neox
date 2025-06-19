#!/usr/bin/env python
"""
Debug the gradient ascent implementation to understand the exact issue.
"""

import torch
import torch.nn as nn


def test_basic_gradient_ascent():
    """Test basic gradient ascent without our custom implementation"""
    print("="*60)
    print("TEST 1: Basic Gradient Ascent (Manual)")
    print("="*60)
    
    # Simple quadratic function: f(x) = x^2
    x = torch.tensor([2.0], requires_grad=True)
    
    # Forward pass
    y = x ** 2
    print(f"f(x) = x² at x = {x.item()}")
    print(f"f(x) = {y.item()}")
    
    # Backward pass
    y.backward()
    print(f"Gradient df/dx = {x.grad.item()}")
    
    # Gradient DESCENT step (minimize f)
    x_new_descent = x.data - 0.5 * x.grad.data
    print(f"\nGradient DESCENT:")
    print(f"x_new = x - 0.5 * grad = {x_new_descent.item()}")
    print(f"f(x_new) = {x_new_descent.item()**2} (should be < {y.item()})")
    
    # Gradient ASCENT step (maximize f)
    x_new_ascent = x.data + 0.5 * x.grad.data
    print(f"\nGradient ASCENT:")
    print(f"x_new = x + 0.5 * grad = {x_new_ascent.item()}")
    print(f"f(x_new) = {x_new_ascent.item()**2} (should be > {y.item()})")


def test_negated_loss_method():
    """Test gradient ascent via negated loss"""
    print("\n" + "="*60)
    print("TEST 2: Gradient Ascent via Negated Loss")
    print("="*60)
    
    # Model: single parameter
    w = torch.tensor([2.0], requires_grad=True)
    
    # Loss function: L(w) = w^2
    loss = w ** 2
    print(f"Loss L(w) = w² at w = {w.item()}")
    print(f"L(w) = {loss.item()}")
    
    # Method 1: Direct gradient ascent
    loss.backward(retain_graph=True)
    grad_direct = w.grad.clone()
    w.grad.zero_()
    print(f"\nMethod 1 - Direct ascent:")
    print(f"Gradient dL/dw = {grad_direct.item()}")
    w_new_1 = w.data + 0.5 * grad_direct  # Ascent
    print(f"w_new = w + 0.5 * grad = {w_new_1.item()}")
    print(f"L(w_new) = {w_new_1.item()**2}")
    
    # Method 2: Negated loss (our approach)
    neg_loss = -loss
    neg_loss.backward()
    grad_negated = w.grad.clone()
    print(f"\nMethod 2 - Negated loss:")
    print(f"Gradient d(-L)/dw = {grad_negated.item()}")
    w_new_2 = w.data - 0.5 * grad_negated  # Descent on -L = Ascent on L
    print(f"w_new = w - 0.5 * grad = {w_new_2.item()}")
    print(f"L(w_new) = {w_new_2.item()**2}")
    
    print(f"\nBoth methods should give same result:")
    print(f"Method 1: w_new = {w_new_1.item()}, L = {w_new_1.item()**2}")
    print(f"Method 2: w_new = {w_new_2.item()}, L = {w_new_2.item()**2}")


def test_our_implementation_logic():
    """Test our actual implementation logic"""
    print("\n" + "="*60)
    print("TEST 3: Our Implementation Logic")
    print("="*60)
    
    # Simulate what happens in our cross entropy
    w = torch.tensor([2.0], requires_grad=True)
    
    # Forward pass with sample_sign = -1
    loss = w ** 2
    sample_sign = -1.0
    loss_modified = loss * sample_sign  # This is what forward does
    
    print(f"Original loss: {loss.item()}")
    print(f"Sample sign: {sample_sign}")
    print(f"Modified loss (forward): {loss_modified.item()}")
    
    # Backward pass
    loss_modified.backward()
    grad_after_backward = w.grad.clone()
    print(f"\nGradient after backward: {grad_after_backward.item()}")
    
    # Our backward pass multiplies by sample_sign again
    grad_final = grad_after_backward * sample_sign
    print(f"Final gradient (after our backward): {grad_final.item()}")
    
    # Update step
    w_new = w.data - 0.5 * grad_final  # Standard optimizer step
    print(f"\nUpdate: w_new = w - 0.5 * grad_final = {w_new.item()}")
    print(f"New loss: {w_new.item()**2}")
    
    print(f"\nAnalysis:")
    print(f"- Original gradient dL/dw = {2*w.item()} = {2*w.item()}")
    print(f"- After forward multiply: gradient of ({sample_sign}*L) = {sample_sign} * {2*w.item()} = {sample_sign * 2*w.item()}")
    print(f"- After backward multiply: {sample_sign} * {sample_sign * 2*w.item()} = {sample_sign**2 * 2*w.item()} = {2*w.item()}")
    print(f"- This is DESCENT not ASCENT!")


def test_correct_implementation():
    """Test the correct implementation"""
    print("\n" + "="*60)
    print("TEST 4: What the Implementation SHOULD Be")
    print("="*60)
    
    print("Option 1: Only modify gradients in backward (cleaner):")
    w = torch.tensor([2.0], requires_grad=True)
    loss = w ** 2
    loss.backward()
    grad = w.grad.clone()
    
    # For ascent, flip the gradient
    grad_ascent = -grad
    w_new = w.data - 0.5 * grad_ascent  # Standard update
    print(f"w: {w.item()} -> {w_new.item()}")
    print(f"loss: {w.item()**2} -> {w_new.item()**2} (increased ✓)")
    
    print("\nOption 2: Keep forward modification, but DON'T multiply in backward:")
    w = torch.tensor([2.0], requires_grad=True)
    sample_sign = -1.0
    loss = w ** 2
    loss_modified = loss * sample_sign
    loss_modified.backward()
    # Don't multiply gradient again!
    w_new = w.data - 0.5 * w.grad
    print(f"w: {w.item()} -> {w_new.item()}")
    print(f"loss: {w.item()**2} -> {w_new.item()**2} (increased ✓)")


def identify_the_bug():
    """Clearly identify the bug"""
    print("\n" + "="*60)
    print("THE BUG IDENTIFIED")
    print("="*60)
    
    print("Current implementation:")
    print("1. Forward: loss = loss * sign (-1 for ascent)")
    print("2. Backward: grad = grad * sign (-1 for ascent)")
    print("3. Result: sign * sign = (-1) * (-1) = 1")
    print("4. This gives GRADIENT DESCENT!")
    
    print("\nThe issue: We're multiplying by the sign TWICE!")
    print("- Once in forward (modifying the loss)")
    print("- Once in backward (modifying the gradient)")
    print("- The double negation cancels out!")
    
    print("\nCorrect implementation should be ONE of:")
    print("A) Modify loss in forward, DON'T modify gradient in backward")
    print("B) DON'T modify loss in forward, modify gradient in backward")
    print("C) Modify both but use |sign| in backward to avoid double negation")


if __name__ == "__main__":
    test_basic_gradient_ascent()
    test_negated_loss_method()
    test_our_implementation_logic()
    test_correct_implementation()
    identify_the_bug()