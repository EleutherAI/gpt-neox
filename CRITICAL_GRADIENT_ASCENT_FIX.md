# Critical Gradient Ascent Bug Fix

## The Bug

The original gradient ascent implementation had a critical flaw that caused it to perform gradient **descent** instead of ascent on dangerous examples, making the model learn them better instead of unlearning them.

### What Was Wrong

1. The forward pass multiplied the loss by `sample_signs` (-1 for ascent samples)
2. The backward pass **completely ignored** `sample_signs` 
3. This resulted in the optimizer minimizing the negated loss
4. Since min(-loss) = max(loss), this was supposed to work, BUT...
5. The backward pass wasn't applying the sign, so gradients were wrong

### The Math

For gradient ascent, we want to maximize the loss L. This is equivalent to minimizing -L.

**What the code was doing:**
- Forward: compute -L (by multiplying L by -1)
- Backward: compute ∇(-L) = -∇L
- Optimizer step: θ = θ - lr * (-∇L) = θ + lr * ∇L ✓ (correct)

**What actually happened:**
- Forward: compute -L (by multiplying L by -1)  
- Backward: compute ∇L (sign was ignored!) ❌
- Optimizer step: θ = θ - lr * ∇L (gradient descent!) ❌

## The Fix

The fix ensures that the backward pass also applies the sample signs:

```python
# In backward pass:
if sample_signs is not None:
    # Apply scaling for gradient ascent samples
    if gradient_ascent_loss_scale != 1.0:
        scaled_signs = sample_signs.clone()
        ascent_mask = sample_signs < 0
        scaled_signs[ascent_mask] = sample_signs[ascent_mask] * gradient_ascent_loss_scale
        grad_input.mul_(scaled_signs.unsqueeze(dim=-1))
    else:
        grad_input.mul_(sample_signs.unsqueeze(dim=-1))
```

Now the math works correctly:
- Forward: compute loss * sample_signs (negative for ascent)
- Backward: compute gradient * sample_signs (double negation for ascent)
- Result: Gradient ascent on dangerous examples

## Impact

This bug meant that all previous gradient ascent training runs were actually making the model **better** at dangerous tasks instead of worse. This explains why benchmark performance improved when it should have degraded.

## Testing

Run `test_gradient_ascent_fix.py` to verify:
1. Gradient descent (signs=1) decreases loss ✓
2. Gradient ascent (signs=-1) increases loss ✓
3. Mixed signs with scaling work correctly ✓