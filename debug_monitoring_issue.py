#!/usr/bin/env python
"""
Debug the actual monitoring metric issue.
"""

import torch

# Simulate more realistic scenario
seq_length = 2048
batch_size = 32
vocab_size = 50304

# Create realistic loss values (4-5 is typical for untrained, ~2 for trained)
token_losses = torch.rand(batch_size, seq_length) * 0.5 + 1.8  # 1.8-2.3 range

# Create gradient signs with 1.21% ascent (like your data)
gradient_signs = torch.ones(batch_size, seq_length)
n_ascent_tokens = int(batch_size * seq_length * 0.0121)
# Randomly mark some tokens as ascent
flat_indices = torch.randperm(batch_size * seq_length)[:n_ascent_tokens]
gradient_signs.view(-1)[flat_indices] = -1

# Loss mask (assume all tokens are unmasked for simplicity)
loss_mask = torch.ones(batch_size, seq_length)

print("=== Debugging Monitoring Metrics ===\n")

# Main loss calculation (from cross_entropy function)
total_loss = (token_losses * loss_mask).sum() / loss_mask.sum()
print(f"Main loss (per-token average): {total_loss:.4f}")

# Current monitoring calculation
masked_losses = token_losses * loss_mask
masked_losses_per_seq = masked_losses.sum(-1) / loss_mask.sum(-1)

# The issue: gradient signs are per-token, but we're averaging per-sequence
ascent_mask = (gradient_signs < 0)  # Shape: [batch, seq_len]
descent_mask = (gradient_signs >= 0)

# Current approach: for each sequence, check if it has ANY ascent tokens
# This is wrong! We want per-token statistics
sequences_with_ascent = ascent_mask.any(dim=1)  # Shape: [batch]
sequences_all_descent = ~sequences_with_ascent

print(f"\nProblem: Gradient signs are per-token, but metric averages per-sequence")
print(f"Total tokens: {batch_size * seq_length}")
print(f"Ascent tokens: {(gradient_signs < 0).sum().item()} ({(gradient_signs < 0).sum().item()/(batch_size * seq_length)*100:.2f}%)")
print(f"Sequences with ANY ascent token: {sequences_with_ascent.sum().item()} / {batch_size}")

# The current code is doing this (WRONG):
if sequences_with_ascent.sum() > 0:
    # Takes average loss of entire sequences that contain ANY ascent token
    ascent_losses_wrong = masked_losses_per_seq[sequences_with_ascent]
    wrong_ascent_metric = ascent_losses_wrong.mean().item()
    print(f"\nCurrent (wrong) ascent metric: {wrong_ascent_metric:.4f}")

# What about sequences that have BOTH ascent and descent tokens?
# They get counted in BOTH metrics!
mixed_sequences = ascent_mask.any(dim=1) & descent_mask.any(dim=1)
print(f"Sequences with BOTH ascent and descent: {mixed_sequences.sum().item()}")

print("\n=== The Real Issue ===")
print("The monitoring code is calculating per-sequence averages, but:")
print("1. It's checking gradient_signs < 0 at the TOKEN level")
print("2. Then taking masked_losses at the SEQUENCE level")
print("3. A sequence with even 1 ascent token out of 2048 gets its ENTIRE")
print("   sequence loss counted as 'ascent loss'")

# Here's what's likely happening:
# Very few sequences have ALL ascent tokens, so the 'ascent loss' is actually
# the average loss of normal sequences that happen to contain a few ascent tokens

# Calculate what the metric SHOULD be
token_losses_flat = token_losses.view(-1)
gradient_signs_flat = gradient_signs.view(-1)
loss_mask_flat = loss_mask.view(-1)

ascent_tokens = (gradient_signs_flat < 0) & (loss_mask_flat > 0)
descent_tokens = (gradient_signs_flat >= 0) & (loss_mask_flat > 0)

correct_ascent_loss = token_losses_flat[ascent_tokens].mean().item() if ascent_tokens.sum() > 0 else 0
correct_descent_loss = token_losses_flat[descent_tokens].mean().item() if descent_tokens.sum() > 0 else 0

print(f"\nCorrect per-token metrics:")
print(f"Ascent loss: {correct_ascent_loss:.4f}")
print(f"Descent loss: {correct_descent_loss:.4f}")
print(f"Main loss: {total_loss:.4f}")
print(f"\nThese should all be similar (within noise)")