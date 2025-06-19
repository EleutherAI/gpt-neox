#!/usr/bin/env python
"""
Debug the backward pass to see what's happening.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def test_autograd_behavior():
    """Test PyTorch's autograd behavior with custom functions"""
    
    print("TESTING AUTOGRAD BEHAVIOR")
    print("=" * 60)
    
    # Create a custom autograd function that mimics our cross entropy
    class TestFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, sign=1.0):
            ctx.sign = sign
            # Simple quadratic loss
            loss = x ** 2
            # Apply sign in forward
            loss_modified = loss * sign
            ctx.save_for_backward(x)
            return loss_modified
        
        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            sign = ctx.sign
            
            # Gradient of x^2 is 2x
            grad_x = 2 * x
            
            # The key question: should we multiply by sign here?
            # NO! PyTorch already handles this because we modified the loss in forward
            
            # Just return the gradient multiplied by incoming gradient
            return grad_x * grad_output, None
    
    # Test 1: Normal (sign = 1)
    print("\n1. Normal case (sign = 1):")
    x = torch.tensor(2.0, requires_grad=True)
    loss = TestFunction.apply(x, 1.0)
    print(f"x = {x.item()}")
    print(f"loss = {loss.item()}")
    
    loss.backward()
    print(f"grad = {x.grad.item()}")
    
    with torch.no_grad():
        x_new = x - 0.1 * x.grad
    print(f"x_new = {x_new.item()}")
    print(f"loss_new = {x_new.item()**2} (should decrease)")
    
    # Test 2: Ascent (sign = -1)
    print("\n2. Ascent case (sign = -1):")
    x = torch.tensor(2.0, requires_grad=True)
    loss = TestFunction.apply(x, -1.0)
    print(f"x = {x.item()}")
    print(f"loss = {loss.item()} (negative)")
    
    loss.backward()
    print(f"grad = {x.grad.item()}")
    
    with torch.no_grad():
        x_new = x - 0.1 * x.grad
    print(f"x_new = {x_new.item()}")
    print(f"loss_new = {x_new.item()**2} (should increase)")
    
    # The issue might be that grad_output already includes the sign!


def check_grad_output_behavior():
    """Check how grad_output behaves"""
    
    print("\n\nCHECKING GRAD_OUTPUT BEHAVIOR")
    print("=" * 60)
    
    class DebugFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x ** 2
        
        @staticmethod
        def backward(ctx, grad_output):
            print(f"grad_output received in backward: {grad_output}")
            return grad_output * 2  # Simplified for testing
    
    # Test what grad_output looks like
    x = torch.tensor(2.0, requires_grad=True)
    
    # Case 1: Direct backward
    print("\nCase 1: Direct backward")
    y = DebugFunction.apply(x)
    print(f"y = {y.item()}")
    y.backward()
    
    # Case 2: Negated loss
    x.grad = None
    print("\nCase 2: Negated loss backward")
    y = DebugFunction.apply(x)
    neg_y = -y
    print(f"neg_y = {neg_y.item()}")
    neg_y.backward()
    
    print("\nKey insight: When we negate the loss, grad_output is negated!")


def trace_cross_entropy_backward():
    """Trace through the actual cross entropy backward"""
    
    print("\n\nTRACING CROSS ENTROPY BACKWARD")
    print("=" * 60)
    
    from megatron.mpu.cross_entropy import _VocabParallelCrossEntropy
    
    # Let me check if the issue is in how gradients flow
    class TracedCE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, target, sample_sign):
            # Simplified CE
            probs = torch.softmax(logits, dim=-1)
            ce = -torch.log(probs[0, target])
            
            # Apply sign
            loss = ce * sample_sign
            
            ctx.save_for_backward(probs, target)
            ctx.sample_sign = sample_sign
            
            print(f"Forward: CE = {ce.item():.4f}, sign = {sample_sign}, loss = {loss.item():.4f}")
            
            return loss
        
        @staticmethod  
        def backward(ctx, grad_output):
            probs, target = ctx.saved_tensors
            sample_sign = ctx.sample_sign
            
            print(f"Backward: grad_output = {grad_output.item():.4f}")
            
            # Standard CE gradient
            grad = probs.clone()
            grad[target] -= 1.0
            
            # Multiply by incoming gradient
            grad = grad * grad_output
            
            print(f"Final gradient: {grad}")
            
            return grad, None, None
    
    # Test
    logits = torch.tensor([2.0, 1.0, 0.0], requires_grad=True)
    target = torch.tensor(0)
    
    print("\nTest 1: Descent (sign = 1)")
    loss = TracedCE.apply(logits, target, 1.0)
    loss.backward()
    print(f"Logits gradient: {logits.grad}")
    
    logits.grad = None
    print("\nTest 2: Ascent (sign = -1)")
    loss = TracedCE.apply(logits, target, -1.0)
    loss.backward()
    print(f"Logits gradient: {logits.grad}")


if __name__ == "__main__":
    test_autograd_behavior()
    check_grad_output_behavior()
    trace_cross_entropy_backward()