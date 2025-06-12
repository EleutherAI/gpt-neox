# Gradient Signs Implementation Summary

This document summarizes the implementation of conditional gradient ascent/descent support in GPT-NeoX with separate loss tracking for monitoring.

## Overview

The implementation adds support for:
1. **Auto-detection** of gradient signs data files
2. **Conditional gradient ascent** for samples marked with gradient_sign = -1.0
3. **Separate loss tracking** for gradient ascent and descent samples
4. **Pipeline parallel support** for gradient signs
5. **Weights & Biases integration** for monitoring

## Key Changes

### 1. Auto-Detection of Gradient Signs Files

**File: `megatron/data/data_utils.py`**
- Added `auto_detect_gradient_signs_path()` function that automatically detects gradient signs files
- If data prefix is `/path/to/data_text_document`, it looks for `/path/to/data_gradient_signs`
- Integrated into dataset building functions

### 2. Dataset Support

**File: `megatron/data/gpt2_dataset.py`**
- Added `gradient_signs_dataset` parameter to GPT2Dataset
- Extracts gradient signs from the dataset and creates a tensor with the same length as the sequence
- Properly handles gradient signs in __getitem__ method

### 3. Batch Processing

**File: `megatron/training.py`**

#### Non-Pipeline Parallel Mode
- Modified `_get_batch()` to handle gradient signs in the batch
- Updated `get_batch()` to include gradient signs in returned values
- Modified `forward_step()` to:
  - Pass gradient signs to the loss function
  - Calculate separate losses for monitoring
  - Track gradient ascent/descent samples and their average losses

#### Pipeline Parallel Mode
- Modified `get_batch_pipe()` to:
  - Extract gradient signs from the batch
  - Store them in `neox_args._current_gradient_signs` for pipeline parallel access
  - Handle curriculum learning adjustments
- Updated `train_step_pipe()` to extract and aggregate additional losses

### 4. Loss Calculation

**File: `megatron/model/gpt2_model.py`**
- Modified `cross_entropy()` to accept `sample_signs` parameter
- Added `cross_entropy_with_gradient_tracking()` for pipeline parallel mode:
  - Extracts gradient signs from neox_args
  - Calculates separate losses for monitoring
  - Stores metrics in neox_args for aggregation
- Updated GPT2ModelPipe to use the gradient tracking loss function when in pipeline parallel mode

### 5. Model Parallel Cross Entropy

**File: `megatron/mpu/cross_entropy.py`**
- Modified `vocab_parallel_cross_entropy()` to accept `sample_signs` parameter
- Updated `_VocabParallelCrossEntropy` autograd function to:
  - Apply gradient reversal for samples with negative signs
  - Store sample signs in context for backward pass
  - Properly handle gradient reversal during backward propagation

### 6. Logging Integration

**File: `megatron/logging.py`**
- The `training_log()` function automatically logs all metrics from loss_dict to W&B
- No modifications needed - gradient ascent/descent metrics are automatically logged

## Usage

### Training with Gradient Signs

1. **Prepare gradient signs data**: Create a binary file with the same indexing as your training data, containing -1.0 for gradient ascent samples and 1.0 for normal samples.

2. **Auto-detection**: Place the gradient signs file with the naming convention:
   - If data is at `/path/to/data_text_document`, gradient signs should be at `/path/to/data_gradient_signs`

3. **Manual specification**: Alternatively, specify the path in your config:
   ```yaml
   "train_gradient_signs_data_paths": ["/path/to/gradient_signs"]
   ```

### Monitoring in Weights & Biases

The following metrics are automatically logged:
- `gradient_ascent_loss`: Average loss for gradient ascent samples
- `gradient_descent_loss`: Average loss for normal (gradient descent) samples
- `gradient_ascent_samples`: Number of gradient ascent samples in the batch
- `gradient_descent_samples`: Number of gradient descent samples in the batch
- `lm_loss`: Overall loss (with gradient signs applied)

## Implementation Details

### Gradient Reversal
- Gradients are reversed by negating them during the backward pass
- This is done in the autograd function to ensure proper gradient flow
- The reversal only affects gradient computation, not the forward loss calculation

### Pipeline Parallel Support
- Gradient signs are passed through `neox_args._current_gradient_signs`
- A special loss function extracts and processes gradient signs in pipeline mode
- Additional losses are aggregated and passed back through `neox_args._additional_losses`

### Memory Efficiency
- Gradient signs are stored as float32 tensors
- Monitoring calculations use `torch.no_grad()` to avoid memory overhead
- Sample-wise loss calculation is done only when gradient signs are present

## Testing

Run the test script to verify the implementation:
```bash
python test_gradient_signs_pipeline.py
```

This tests:
1. Gradient tracking loss function
2. Auto-detection of gradient signs files
3. Pipeline parallel compatibility