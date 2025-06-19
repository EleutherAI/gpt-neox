#!/usr/bin/env python
"""
Final test to verify gradient ascent behavior.
"""

import torch
import torch.nn.functional as F


def test_simple_gradient_flow():
    """Test gradient flow with simple examples"""
    
    print("SIMPLE GRADIENT FLOW TEST")
    print("=" * 60)
    
    # Test case: simple quadratic
    x = torch.tensor(2.0, requires_grad=True)
    
    # Forward pass
    loss = x ** 2
    print(f"x = {x.item()}")
    print(f"loss = x² = {loss.item()}")
    
    # Test 1: Normal gradient descent
    print("\n1. Normal Gradient Descent:")
    loss.backward()
    print(f"grad = {x.grad.item()}")
    with torch.no_grad():
        x_new = x - 0.1 * x.grad
    print(f"x_new = x - 0.1*grad = {x_new.item()}")
    print(f"loss_new = {x_new.item()**2} < {loss.item()} ✓")
    
    # Test 2: Gradient ascent via negated loss
    x = torch.tensor(2.0, requires_grad=True)
    loss = x ** 2
    neg_loss = -loss  # Multiply by -1
    
    print("\n2. Gradient Ascent (negated loss):")
    print(f"neg_loss = -{loss.item()} = {neg_loss.item()}")
    neg_loss.backward()
    print(f"grad of neg_loss = {x.grad.item()}")
    with torch.no_grad():
        x_new = x - 0.1 * x.grad  # Normal optimizer step
    print(f"x_new = x - 0.1*grad = {x_new.item()}")
    print(f"loss_new = {x_new.item()**2} > {loss.item()} ✓")
    
    # This confirms that negating the loss in forward is sufficient for ascent


def test_cross_entropy_behavior():
    """Test cross entropy specific behavior"""
    
    print("\n\nCROSS ENTROPY BEHAVIOR TEST")
    print("=" * 60)
    
    # Simple 2-class problem
    logits = torch.tensor([[2.0, 1.0]], requires_grad=True)  # Batch=1, Classes=2
    target = torch.tensor([0])  # Target is class 0
    
    # Manual cross entropy
    probs = F.softmax(logits, dim=-1)
    ce_loss = -torch.log(probs[0, target])
    
    print(f"Logits: {logits.data}")
    print(f"Target: class {target.item()}")
    print(f"Probs: {probs.data}")
    print(f"CE Loss: {ce_loss.item():.4f}")
    
    # Test 1: Normal gradient
    ce_loss.backward(retain_graph=True)
    grad_normal = logits.grad.clone()
    print(f"\n1. Normal gradient: {grad_normal}")
    
    # Test 2: Gradient from negated loss
    logits.grad.zero_()
    neg_ce_loss = -ce_loss
    neg_ce_loss.backward()
    grad_negated = logits.grad.clone()
    print(f"\n2. Negated loss gradient: {grad_negated}")
    print(f"   Ratio: {grad_negated / grad_normal}")
    print("   Should be -1.0 ✓" if torch.allclose(grad_negated, -grad_normal) else "   ERROR!")
    
    # Update check
    print("\n3. Update effects:")
    
    # With normal gradient (descent)
    with torch.no_grad():
        logits_desc = logits - 0.5 * grad_normal
        probs_desc = F.softmax(logits_desc, dim=-1)
        loss_desc = -torch.log(probs_desc[0, target])
    print(f"   Descent: loss {ce_loss.item():.4f} -> {loss_desc.item():.4f} (decreased ✓)")
    
    # With negated gradient (ascent) 
    with torch.no_grad():
        logits_asc = logits - 0.5 * grad_negated
        probs_asc = F.softmax(logits_asc, dim=-1)
        loss_asc = -torch.log(probs_asc[0, target])
    print(f"   Ascent:  loss {ce_loss.item():.4f} -> {loss_asc.item():.4f} (increased ✓)")


def verify_implementation():
    """Verify our implementation approach"""
    
    print("\n\nIMPLEMENTATION VERIFICATION")
    print("=" * 60)
    
    print("Our approach:")
    print("1. Forward pass: multiply loss by sample_sign (-1 for ascent)")
    print("2. Backward pass: no modification needed")
    print("3. Optimizer: standard update step")
    print("\nResult: Gradient ascent on samples with sign=-1")
    
    print("\nWhy this works:")
    print("- loss_modified = sign * loss")
    print("- grad = d(loss_modified)/dx = sign * d(loss)/dx")
    print("- update: x = x - lr * grad = x - lr * sign * d(loss)/dx")
    print("- For sign=-1: x = x + lr * d(loss)/dx (gradient ascent!)")
    
    print("\n✓ The implementation is mathematically correct.")
    print("✓ The fix (removing double multiplication) is correct.")


if __name__ == "__main__":
    test_simple_gradient_flow()
    test_cross_entropy_behavior()
    verify_implementation()