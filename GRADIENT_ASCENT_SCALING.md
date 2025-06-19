# Gradient Ascent Loss Scaling Implementation

## Overview

This implementation adds a `gradient_ascent_loss_scale` parameter to GPT-NeoX that amplifies the gradient ascent effect for samples marked with negative gradient signs. This is particularly useful when you have imbalanced data with very few gradient ascent samples.

## Problem Solved

In your case, only 1.21% of training samples were marked for gradient ascent, making the effect negligible. The model continued to improve on all samples because gradient descent on 98.79% of data overwhelmed the gradient ascent on 1.21%.

## Implementation Details

### 1. New Configuration Parameter

Added `gradient_ascent_loss_scale` to NeoXArgs:
- Location: `megatron/neox_arguments/neox_args.py`
- Default value: 1.0 (no scaling)
- Type: float

### 2. Updated Functions

- `vocab_parallel_cross_entropy()` in `megatron/mpu/cross_entropy.py`
  - Now accepts `gradient_ascent_loss_scale` parameter
  - Scales the loss for samples where `gradient_signs < 0`

- `cross_entropy()` in `megatron/model/gpt2_model.py`
  - Passes through the scaling parameter

- `forward_step()` in `megatron/training.py`
  - Uses the scaling parameter from neox_args

### 3. Pipeline Parallel Support

The pipeline parallel mode automatically uses the scaled loss through the `cross_entropy_with_gradient_tracking` function, which reads the scale from neox_args.

## How It Works

When `gradient_ascent_loss_scale > 1.0`:
- Descent samples (gradient_signs = 1.0): loss unchanged
- Ascent samples (gradient_signs = -1.0): loss multiplied by `-gradient_ascent_loss_scale`

Example with scale = 10.0:
- Descent sample: loss = 2.5 → final loss = 2.5 × 1.0 = 2.5
- Ascent sample: loss = 2.5 → final loss = 2.5 × (-10.0) = -25.0

This makes the gradient ascent effect 10x stronger for those samples.

## Usage Example

```yaml
# In your config file
"train_gradient_signs_data_paths": ["/path/to/gradient_signs"],
"gradient_ascent_loss_scale": 50.0,  # 50x amplification
```

## Recommended Scaling Values

Based on your data (1.21% ascent samples):
- **Testing**: 10.0 - Mild effect, good for initial testing
- **Moderate**: 25.0 - Noticeable effect
- **Strong**: 50.0 - Significant effect (recommended)
- **Balanced**: 82.6 - Mathematically balanced (1/0.0121)
- **Aggressive**: 100.0+ - Use with caution, may cause instability

## Monitoring

Watch these metrics in W&B:
- `train/gradient_ascent_loss`: Should INCREASE during training
- `train/gradient_descent_loss`: Should DECREASE normally
- `train/lm_loss`: Overall loss (affected by scaling)

If gradient_ascent_loss decreases, the scaling is too weak.

## Quick Test

Before full training, run:
```bash
python test_gradient_ascent_simple.py
```

This verifies the scaling logic and estimates the effect.

## Example Configurations

Two example configs are provided:
1. `configs/gradient_ascent_example.yml` - General example with documentation
2. `configs/annealing_gradient_ascent_scaled.yml` - Specific to your use case

## Troubleshooting

1. **Training unstable**: Reduce `gradient_ascent_loss_scale` or learning rate
2. **No effect visible**: Increase `gradient_ascent_loss_scale` 
3. **Loss goes negative**: This is expected with strong scaling, optimizer handles it
4. **Memory issues**: The scaling doesn't increase memory usage

## Next Steps

1. Start with `gradient_ascent_loss_scale: 10.0` for a test run
2. Monitor gradient_ascent_loss - it should increase
3. Gradually increase the scale until you achieve the desired effect
4. Run your benchmark to verify the model performs worse on target samples