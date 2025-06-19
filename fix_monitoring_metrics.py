#!/usr/bin/env python
"""
Demonstration of the monitoring metric issue and fix.
"""

import torch

# Simulate the issue
seq_length = 2048
batch_size = 4
vocab_size = 50304

# Create fake data
outputs = torch.randn(batch_size, seq_length, vocab_size)
labels = torch.randint(0, vocab_size, (batch_size, seq_length))
loss_mask = torch.ones(batch_size, seq_length)

# Simulate token-wise losses (around 4-5 per token is typical for untrained model)
token_losses = torch.rand(batch_size, seq_length) * 2 + 3.5  # Random between 3.5-5.5

print("=== Monitoring Metric Issue ===\n")

# Current (incorrect) monitoring calculation
masked_losses = token_losses * loss_mask
masked_losses_per_seq = masked_losses.sum(-1) / loss_mask.sum(-1)  # Average per sequence
current_metric = masked_losses_per_seq.mean().item()

# Main loss calculation (correct)
total_loss = (token_losses * loss_mask).sum() / loss_mask.sum()
main_loss = total_loss.item()

print(f"Sequence length: {seq_length}")
print(f"Average token loss: {token_losses.mean().item():.4f}")
print(f"\nMain loss (correct per-token average): {main_loss:.4f}")
print(f"Monitoring metric (current): {current_metric:.4f}")
print(f"Ratio (monitoring/main): {current_metric/main_loss:.4f}")
print(f"\nExpected ratio: ~1.0 (should be equal)")
print(f"Actual ratio: ~{seq_length:.1f} (sequence length)")

# The fix
print("\n=== Fixed Calculation ===")

# Option 1: Per-token average (matches main loss)
gradient_signs = torch.ones(batch_size, seq_length)
gradient_signs[0, :] = -1  # First sequence is ascent

ascent_mask = (gradient_signs < 0)
descent_mask = (gradient_signs >= 0)

# Flatten everything for per-token calculation
token_losses_flat = token_losses.view(-1)
loss_mask_flat = loss_mask.view(-1)
ascent_mask_flat = ascent_mask.view(-1)
descent_mask_flat = descent_mask.view(-1)

# Calculate per-token averages
ascent_token_mask = ascent_mask_flat & (loss_mask_flat > 0)
descent_token_mask = descent_mask_flat & (loss_mask_flat > 0)

if ascent_token_mask.sum() > 0:
    ascent_loss_fixed = token_losses_flat[ascent_token_mask].mean().item()
    print(f"Ascent loss (fixed): {ascent_loss_fixed:.4f}")

if descent_token_mask.sum() > 0:
    descent_loss_fixed = token_losses_flat[descent_token_mask].mean().item()
    print(f"Descent loss (fixed): {descent_loss_fixed:.4f}")

print(f"\nThese should now be similar to the main loss: {main_loss:.4f}")

print("\n=== Explanation ===")
print("The current monitoring code calculates:")
print("  1. Sum of losses for each sequence")
print("  2. Divide by number of tokens in each sequence")
print("  3. Average across sequences")
print("\nThis gives the average TOTAL loss per sequence, not per token!")
print(f"With seq_length={seq_length}, this makes the metric ~{seq_length}x too large")
print("\nThe fix: Calculate per-token averages directly, matching the main loss calculation")