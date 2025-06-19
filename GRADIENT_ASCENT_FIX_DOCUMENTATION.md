# Gradient Ascent Fix Documentation

## Overview
This document details the changes made to fix a critical bug in GPT-NeoX's gradient ascent implementation that was causing the model to learn (gradient descent) instead of unlearn (gradient ascent) on dangerous content.

## The Bug
The original implementation was multiplying by `sample_signs` in BOTH the forward and backward passes, causing double negation that resulted in gradient descent instead of ascent.

## Changes Made

### 1. Core Fix: `/workspace/local_repos/gpt-neox/megatron/mpu/cross_entropy.py`

**Location:** `_VocabParallelCrossEntropy` class, `backward()` method

**Original Code (BUGGY):**
```python
@staticmethod
def backward(ctx, grad_output):
    # ... existing code ...
    if ctx.sample_signs is not None:
        # BUG: This multiplication causes double negation!
        grad_input.mul_(sample_signs.unsqueeze(dim=-1))
```

**Fixed Code:**
```python
@staticmethod
def backward(ctx, grad_output):
    # ... existing code ...
    # REMOVED the multiplication by sample_signs in backward pass
    # This fixes the double negation bug
```

**Why This Fixes It:**
- Forward pass: `loss = loss * sample_signs` (when sample_signs = -1, negates loss)
- Backward pass: No longer multiplies by sample_signs
- Result: Single negation causes gradient ascent (maximizes loss)

### 2. Added Gradient Ascent Scaling

**Location:** `_VocabParallelCrossEntropy` class, `forward()` method

**Added Feature:**
```python
if sample_signs is not None:
    if gradient_ascent_loss_scale != 1.0:
        # Scale the gradient ascent samples more aggressively
        scaled_signs = sample_signs.clone()
        ascent_mask = sample_signs < 0
        scaled_signs[ascent_mask] = sample_signs[ascent_mask] * gradient_ascent_loss_scale
        loss = loss * scaled_signs
    else:
        loss = loss * sample_signs
```

**Purpose:** Amplifies the gradient ascent effect when there are few ascent samples (e.g., 1.21% of dataset).

### 3. Configuration Parameter: `/workspace/local_repos/gpt-neox/megatron/neox_arguments/neox_args.py`

**Added:**
```python
gradient_ascent_loss_scale: float = 1.0
"""
Scaling factor for gradient ascent loss. When gradient signs are -1 (ascent), 
the loss will be multiplied by this factor. This can be used to amplify the 
gradient ascent effect when there are few ascent samples.
"""
```

### 4. Fixed Monitoring Metrics: `/workspace/local_repos/gpt-neox/megatron/training.py`

**Issue:** Metrics were showing ~30x smaller values than actual loss due to per-sequence instead of per-token calculation.

**Fix:**
```python
# Changed from per-sequence to per-token calculation
sample_losses_flat = sample_losses.view(-1)
loss_mask_flat = loss_mask.view(-1)
gradient_signs_flat = gradient_signs.view(-1)

# Create masks for valid tokens
valid_mask = loss_mask_flat > 0

# Separate ascent and descent tokens (not sequences!)
ascent_mask = (gradient_signs_flat < 0) & valid_mask
descent_mask = (gradient_signs_flat >= 0) & valid_mask

# Calculate per-token losses
ascent_losses = sample_losses_flat[ascent_mask]
descent_losses = sample_losses_flat[descent_mask]
```

### 5. Padding Mismatch Handling: `/workspace/local_repos/gpt-neox/megatron/model/gpt2_model.py`

**Issue:** Gradient signs tensor was smaller than loss tensor (65504 vs 65536) due to padding differences.

**Fix:**
```python
# Handle potential size mismatch due to padding
min_size = min(sample_losses_flat.size(0), loss_mask_flat.size(0), sample_signs_flat.size(0))
sample_losses_flat = sample_losses_flat[:min_size]
loss_mask_flat = loss_mask_flat[:min_size]
sample_signs_flat = sample_signs_flat[:min_size]
```

### 6. Updated Cross Entropy Function Signatures

**Files Modified:**
- `/workspace/local_repos/gpt-neox/megatron/mpu/cross_entropy.py`
- `/workspace/local_repos/gpt-neox/megatron/model/gpt2_model.py`

**Change:** Added `gradient_ascent_loss_scale` parameter to:
- `vocab_parallel_cross_entropy()` function
- `_VocabParallelCrossEntropy.forward()` method
- `_VocabParallelCrossEntropy.backward()` method (stored in context)

## Testing

### Test Files Created:
1. `test_gradient_ascent_fix.py` - Comprehensive test of gradient ascent behavior
2. `verify_final.py` - Simple verification of mathematical correctness
3. `analyze_wandb_run.py` - W&B run analysis tool

### Key Test Results:
- Gradient descent: Loss decreases ✓
- Gradient ascent: Loss increases ✓
- Mixed signs with scaling: Works correctly ✓

## Configuration Example

```yaml
# Example config with gradient ascent
gradient_ascent_loss_scale: 10.0  # Amplify ascent effect 10x
train_gradient_signs_data_paths:
  - /path/to/gradient_signs
```

## Impact

### Before Fix:
- Gradient ascent was actually doing gradient descent
- Models were learning dangerous content better instead of unlearning
- Benchmarks measuring dangerous capabilities improved after training

### After Fix:
- Gradient ascent properly increases loss on dangerous samples
- Models unlearn targeted content
- Benchmarks measuring dangerous capabilities should degrade

## Verification Checklist

When analyzing W&B runs to verify the fix:

1. **gradient_ascent_loss** should INCREASE over training
2. **gradient_descent_loss** should DECREASE over training
3. Losses should diverge (move in opposite directions)
4. Check that gradient_ascent_loss_scale is configured
5. Monitor the percentage of ascent vs descent samples

## Known Issues

1. **Low ascent sample percentage**: With only 1.21% ascent samples, even with scaling, the effect can be overwhelmed by the 98.79% descent samples.

2. **Optimal scaling value**: Finding the right gradient_ascent_loss_scale requires experimentation. Too high can destabilize training.

3. **Metric visibility**: The per-token calculation fix ensures metrics accurately reflect the gradient behavior.

## Deployment Notes

To use the fix:
1. Ensure the latest code with the backward pass fix is deployed
2. Set `gradient_ascent_loss_scale` in your config (recommended: 10-50x)
3. Monitor gradient_ascent_loss to ensure it's increasing
4. Adjust scaling if needed based on results