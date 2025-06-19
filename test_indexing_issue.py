#!/usr/bin/env python
"""
Test what happens with the current indexing approach.
"""

import torch

batch_size = 4
seq_length = 10

# Simulate the current code
sample_losses = torch.rand(batch_size, seq_length) * 2 + 3  # Per-token losses
loss_mask = torch.ones(batch_size, seq_length)
gradient_signs = torch.ones(batch_size, seq_length)
gradient_signs[0, :3] = -1  # Some tokens in first sequence are ascent
gradient_signs[2, 5:] = -1  # Some tokens in third sequence are ascent

print("=== Current Code Behavior ===\n")

# Current calculation
masked_losses = sample_losses * loss_mask
masked_losses_per_seq = masked_losses.sum(-1) / loss_mask.sum(-1)  # Shape: [4]
print(f"masked_losses_per_seq shape: {masked_losses_per_seq.shape}")
print(f"masked_losses_per_seq: {masked_losses_per_seq}")

# Create masks
ascent_mask = (gradient_signs < 0)  # Shape: [4, 10]
descent_mask = (gradient_signs >= 0)  # Shape: [4, 10]
print(f"\nascent_mask shape: {ascent_mask.shape}")
print(f"Sequences with ascent tokens: {ascent_mask.any(dim=1)}")

# What happens when we try to index?
try:
    # This is what the current code tries to do
    ascent_losses = masked_losses_per_seq[ascent_mask]
    print(f"\nIndexing [4] tensor with [4,10] mask...")
    print(f"Result shape: {ascent_losses.shape}")
    print(f"This selects: {len(ascent_losses)} values")
except Exception as e:
    print(f"\nError: {e}")

# The actual behavior: it's using ascent_mask as a boolean mask
# But ascent_mask is 2D, so PyTorch flattens masked_losses_per_seq and repeats it!

# What the code SHOULD be doing:
print("\n=== What Should Happen ===")

# Option 1: Per-sequence metrics (if any token in sequence is ascent)
sequences_with_ascent = ascent_mask.any(dim=1)  # Shape: [4]
sequences_with_descent = descent_mask.any(dim=1)  # Shape: [4]

if sequences_with_ascent.sum() > 0:
    ascent_seq_losses = masked_losses_per_seq[sequences_with_ascent]
    print(f"Sequences with ascent tokens: {sequences_with_ascent.sum()}")
    print(f"Average loss of those sequences: {ascent_seq_losses.mean():.4f}")

# Option 2: Per-token metrics (correct approach)
print("\n=== Correct Per-Token Approach ===")
token_losses_flat = sample_losses.view(-1)
gradient_signs_flat = gradient_signs.view(-1)
loss_mask_flat = loss_mask.view(-1)

ascent_token_mask = (gradient_signs_flat < 0) & (loss_mask_flat > 0)
descent_token_mask = (gradient_signs_flat >= 0) & (loss_mask_flat > 0)

ascent_loss = token_losses_flat[ascent_token_mask].mean()
descent_loss = token_losses_flat[descent_token_mask].mean()
overall_loss = (token_losses_flat * loss_mask_flat).sum() / loss_mask_flat.sum()

print(f"Ascent tokens: {ascent_token_mask.sum()} / {loss_mask_flat.sum()}")
print(f"Ascent loss (per-token): {ascent_loss:.4f}")
print(f"Descent loss (per-token): {descent_loss:.4f}")
print(f"Overall loss: {overall_loss:.4f}")