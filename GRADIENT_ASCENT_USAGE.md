# Gradient Ascent Implementation in GPT-NeoX

This document explains how to use the conditional gradient ascent feature that has been implemented in GPT-NeoX.

## Overview

Conditional gradient ascent allows you to train models where certain training samples contribute with inverted gradients (gradient ascent) while others contribute normally (gradient descent). This is useful for safety training where you want the model to avoid generating dangerous content while still learning from safe content.

## Features Implemented

1. **Data Preprocessing**: Extracts `gradient_sign` fields from JSONL training data
2. **Dataset Loading**: GPT2Dataset returns gradient signs alongside training data
3. **Loss Calculation**: Cross-entropy loss applies per-sample gradient signs
4. **Training Integration**: Forward pass propagates gradient signs to loss calculation
5. **Configuration**: YAML config support for gradient signs data paths
6. **Monitoring**: Separate tracking of gradient ascent vs descent losses

## Data Format

Your training data should be in JSONL format with optional `gradient_sign` fields:

```jsonl
{"text": "This is safe content", "gradient_sign": 1.0}
{"text": "This is dangerous content", "gradient_sign": -1.0}
{"text": "Default safe content"}
```

**Important**: 
- `gradient_sign: 1.0` = Normal gradient descent (learn to predict)
- `gradient_sign: -1.0` = Gradient ascent (learn to avoid predicting)
- If omitted, defaults to `1.0` (gradient descent)

## Data Preparation Pipeline

### Step 1: Generate JSONL with Gradient Signs

Use the filtering pipeline from `filtering_for_danger` with the `--keep-filtered-for-gradient-ascent` flag:

```bash
python download_filtered_dataset.py \
    --base-dataset-path /path/to/base/dataset \
    --filter-results-path /path/to/filter/results \
    --keep-filtered-for-gradient-ascent \
    --output-dir /path/to/output \
    --output-filename gradient_ascent_dataset.jsonl
```

This will generate a JSONL file where:
- Safe documents have `gradient_sign: 1.0`
- Dangerous documents have `gradient_sign: -1.0`

### Step 2: Preprocess Data

Use the modified preprocessing script to tokenize and extract gradient signs:

**For datasets WITH gradient signs:**
```bash
python tools/datasets/preprocess_data.py \
    --input /path/to/gradient_ascent_dataset.jsonl \
    --output-prefix /path/to/processed/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --save-gradient-signs
```

**For legacy datasets WITHOUT gradient signs:**
```bash
python tools/datasets/preprocess_data.py \
    --input /path/to/legacy_dataset.jsonl \
    --output-prefix /path/to/processed/data \
    --vocab-file /path/to/vocab.json \
    --merge-file /path/to/merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod
```

**With gradient signs enabled**, this will create:
- `data_text_document.bin` and `data_text_document.idx` (tokenized text)
- `data_gradient_signs.bin` and `data_gradient_signs.idx` (gradient signs)

**Without gradient signs enabled**, this will only create:
- `data_text_document.bin` and `data_text_document.idx` (tokenized text)
- All documents will be treated as normal gradient descent during training

## Configuration

### For Gradient Ascent Training

Add the gradient signs data path to your training configuration YAML:

```yaml
# Gradient ascent configuration
train_data_paths: ["/path/to/processed/data_text_document"]
train_gradient_signs_data_paths: ["/path/to/processed/data_gradient_signs"]

# Other training settings
seq_length: 2048
train_batch_size: 8
train_micro_batch_size_per_gpu: 1
```

### For Legacy Training (Backward Compatible)

Existing configurations continue to work unchanged:

```yaml
# Legacy configuration - no changes needed
train_data_paths: ["/path/to/legacy/data_text_document"]
# No gradient signs data paths needed

# Same training settings
seq_length: 2048
train_batch_size: 8
train_micro_batch_size_per_gpu: 1
```

**Important**: If you don't specify `train_gradient_signs_data_paths`, all documents will use normal gradient descent (gradient_sign = 1.0), exactly as before.

## Training

Start training normally with your configuration:

```bash
python deepy.py train.py your_config.yml
```

The training will automatically:
1. Load gradient signs alongside text data
2. Apply gradient ascent to samples with `gradient_sign: -1.0`
3. Apply gradient descent to samples with `gradient_sign: 1.0`
4. Log separate losses for monitoring

## Monitoring

During training, you'll see additional metrics:

```
gradient_ascent_loss: 2.45      # Loss for dangerous content (before sign flip)
gradient_ascent_samples: 128    # Number of ascent samples in batch
gradient_descent_loss: 2.89     # Loss for safe content  
gradient_descent_samples: 384   # Number of descent samples in batch
```

These metrics help you understand:
- How well the model is learning to avoid dangerous content (ascent loss should increase)
- How well the model is learning safe content (descent loss should decrease)
- The ratio of dangerous to safe content in each batch

## Example Training Configuration

```yaml
# Model architecture
hidden_size: 2048
num_layers: 24
num_attention_heads: 16
seq_length: 2048

# Data paths
train_data_paths: ["/data/processed/safe_and_dangerous_text_document"]
train_gradient_signs_data_paths: ["/data/processed/safe_and_dangerous_gradient_signs"]

# Training settings
train_batch_size: 32
train_micro_batch_size_per_gpu: 2
gradient_accumulation_steps: 16
train_iters: 10000

# Learning rate
lr: 1.5e-4
min_lr: 1.5e-5
lr_decay_style: "cosine"
warmup: 0.01

# Monitoring
log_interval: 10
save_interval: 1000
eval_interval: 1000

# Gradient ascent monitoring will be automatically enabled
wandb_project: "gradient_ascent_safety_training"
```

## Expected Behavior

With gradient ascent training:

1. **Dangerous content loss increases**: The model learns to assign low probability to dangerous text
2. **Safe content loss decreases**: The model learns to predict safe text normally
3. **Overall capability preserved**: The model maintains general language abilities on safe content
4. **Safety improved**: The model is less likely to generate dangerous content

## Troubleshooting

### Common Issues

1. **"gradient_signs not found in data"**: Ensure your JSONL contains `gradient_sign` fields and preprocessing captured them correctly

2. **Loss doesn't change as expected**: Verify that your gradient signs are correct (-1.0 for ascent, 1.0 for descent)

3. **Memory issues**: Gradient ascent uses slightly more memory due to additional loss calculations for monitoring

4. **Convergence problems**: Balance the ratio of ascent vs descent samples (typically 10-30% ascent)

### Verification

To verify the implementation is working:

1. Check that both `gradient_ascent_loss` and `gradient_descent_loss` appear in logs
2. Confirm ascent loss increases while descent loss decreases over training
3. Test generation to ensure dangerous content is less likely

## Integration with Safety Pipelines

This gradient ascent implementation is designed to work with:

1. **WMDP filtering**: Use WMDP data as ascent samples
2. **Custom safety filters**: Any BERT-based or keyword-based filtering
3. **Human annotation**: Manually labeled dangerous vs safe content
4. **Reinforcement learning**: As a pre-training step before RLHF

The key is generating training data with appropriate `gradient_sign` annotations based on your safety criteria.