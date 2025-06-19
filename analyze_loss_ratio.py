#!/usr/bin/env python
"""
Analyze the exact ratio between main loss and monitoring metrics.
"""

# Your observed values
main_loss = 1.96
monitoring_loss = 0.06

ratio = monitoring_loss / main_loss
print(f"Observed ratio: {monitoring_loss}/{main_loss} = {ratio:.4f}")
print(f"Inverse ratio: {main_loss}/{monitoring_loss} = {main_loss/monitoring_loss:.1f}")

# What could cause a ~30x difference?
print(f"\nPossible sequence lengths that would cause this ratio:")
for seq_len in [32, 64, 128, 256, 512, 1024, 2048]:
    expected_ratio = 1.0 / seq_len
    if abs(expected_ratio - ratio) < 0.01:
        print(f"  Sequence length {seq_len}: ratio = {expected_ratio:.4f} ✓")
    else:
        print(f"  Sequence length {seq_len}: ratio = {expected_ratio:.4f}")

# Another possibility: effective sequence length due to masking
print(f"\nIf sequences have padding:")
for effective_len in [30, 32, 35, 40, 50, 60, 64]:
    expected_ratio = 1.0 / effective_len
    if abs(expected_ratio - ratio) < 0.005:
        print(f"  Effective length ~{effective_len}: ratio = {expected_ratio:.4f} ✓")

# Or maybe it's not sequence length at all
print(f"\nActual issue found:")
print(f"The old code was calculating:")
print(f"1. Per-token losses")
print(f"2. Sum over sequence, divide by sequence length = average per-token loss")
print(f"3. But then comparing this to sequences that have ANY ascent token")
print(f"4. With 1.21% ascent, every sequence has both types")
print(f"5. So both metrics measure the same thing: average loss across all sequences")
print(f"\nThe 30x difference suggests the loss mask is removing ~97% of tokens")
print(f"Or there's another aggregation step happening somewhere")