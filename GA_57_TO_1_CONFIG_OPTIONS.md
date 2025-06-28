# Configuration Options for 57:1 GA to GD Ratio

## Current Limitation

The current "interleaved" mode only supports doing 1 GA step at a time, alternating with GD steps. It cannot do multiple GA iterations in a row.

## Option 1: Use "interval" Mode (Recommended)

The existing "interval" mode can already do what you want. Configure it like this:

```yaml
# Gradient Ascent - Interval Mode with 57:1 ratio
"ga_dataset": "/data/ga_gold_dataset/ga_gold_dataset_text_document",
"ga_dataset_impl": "mmap",
"ga_mode": "interval",      # Use interval mode instead of interleaved
"ga_interval": 58,           # Do GA every 58 iterations (57 GA + 1 GD = 58 total)
"ga_iters": 57,              # Do 57 GA iterations each time
"ga_lr_scale": 0.5
```

This will:
- Do 1 GD iteration
- Then do 57 GA iterations in a burst
- Repeat this pattern

**Pros:**
- Works with existing code
- No modifications needed
- Clear separation between GD and GA phases

**Cons:**
- Not truly "interleaved" - it's burst mode
- All 57 GA iterations happen consecutively

## Option 2: Modify the Code for True Interleaving

If you want true interleaving (GA, GA, GA, ...[57 times], GD, GA, GA, GA...), we'd need to modify the training loop.

Here's the modification needed in `training.py`:

```python
# Around line 1551-1558, replace the interleaved logic with:
elif neox_args.ga_mode == "interleaved":
    # Modified for N:1 GA:GD ratio
    # For ratio R, do R consecutive GA steps, then 1 GD step
    cycle_length = neox_args.ga_interleave_ratio + 1
    position_in_cycle = iteration % cycle_length
    
    # Do GA for first R positions in cycle, GD for last position
    do_ga = (iteration > 0 and position_in_cycle < neox_args.ga_interleave_ratio)
```

Then set `ga_interleave_ratio: 57` in the config.

**Pros:**
- True interleaving pattern
- Each iteration is either GA or GD (not bursts)

**Cons:**
- Requires code modification
- More complex to implement and debug

## Option 3: Create a New "Burst-Interleaved" Mode

Add a new mode that combines the burst capability of interval mode with the regular pattern of interleaved mode:

```yaml
"ga_mode": "burst_interleaved",  # New mode
"ga_interleave_ratio": 1,        # After every 1 GD step
"ga_iters": 57,                  # Do 57 GA iterations
```

This would require adding new logic to handle this mode.

## Recommendation

**Use Option 1** with interval mode. It's the simplest and requires no code changes. The config would be:

```yaml
{
  # ... other settings ...
  
  # Gradient Ascent - 57:1 GA:GD ratio
  "ga_dataset": "/data/ga_gold_dataset/ga_gold_dataset_text_document",
  "ga_dataset_impl": "mmap",
  "ga_mode": "interval",
  "ga_interval": 58,      # Every 58 iterations
  "ga_iters": 57,         # Do 57 GA iterations
  "ga_lr_scale": 0.5
}
```

## Important Considerations

1. **Training Time**: This will make training much slower (57x more GA steps)
2. **Learning Rate**: You might need to adjust `ga_lr_scale` - with 57 consecutive GA steps, 0.5 might be too aggressive
3. **Stability**: 57 consecutive GA steps might cause instability - monitor carefully
4. **Checkpointing**: Consider saving checkpoints more frequently to track the effect

## Why This Might Help

Based on the early trend analysis showing GA works initially but fails later:
- 57 consecutive GA steps might overcome the model's resistance
- The concentrated "attack" might prevent the model from recovering
- Could maintain the unlearning effect longer into training

However, this is quite extreme - typical GA ratios in research are 1:1 to 1:10, not 57:1.