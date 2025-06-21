# Gradient Ascent for Machine Unlearning in GPT-NeoX

## Overview

The Gradient Ascent (GA) feature in GPT-NeoX enables selective knowledge removal through machine unlearning. By maximizing loss on specific datasets during training, models can "forget" unwanted information while preserving general capabilities. This is particularly useful for:

- Removing copyrighted or sensitive content
- Eliminating harmful or biased knowledge
- Complying with data removal requests (e.g., GDPR)
- Reducing dangerous capabilities in models

## How It Works

### Core Concept

While standard training uses gradient descent to minimize loss:
```
θ_{t+1} = θ_t - η∇L(x, θ_t)  # Minimize loss (learn)
```

Gradient ascent reverses this to maximize loss:
```
θ_{t+1} = θ_t + η∇L(x, θ_t)  # Maximize loss (unlearn)
```

In practice, we implement this by negating the loss before backpropagation, causing the optimizer to increase rather than decrease the loss on the forget dataset.

### Training Process

The GA implementation interleaves unlearning with regular training:

1. **Regular Training**: Model learns from main dataset (gradient descent)
2. **GA Trigger**: At specified intervals, switch to forget dataset
3. **Unlearning**: Perform gradient ascent for configured iterations
4. **Resume**: Return to regular training

This interleaved approach maintains model utility while selectively removing unwanted knowledge.

## Configuration

### Required Parameters

Add these parameters to your NeoX configuration file:

```yaml
# Path to the dataset containing content to unlearn
ga_dataset: "/path/to/forget/dataset"

# Mode for gradient ascent execution
ga_mode: "interval"  # or "interleaved"

# For interval mode (original behavior)
ga_interval: 100     # Trigger GA every N iterations
ga_iters: 5         # Number of GA iterations per trigger

# For interleaved mode (new behavior)
ga_interleave_ratio: 1  # Ratio of GD:GA batches (1 = 1:1, 2 = 2:1, etc.)

# Learning rate scaling factor for GA (recommended: 2.0-5.0)
ga_lr_scale: 3.0

# Dataset implementation (optional, defaults to "mmap")
ga_dataset_impl: "mmap"
```

### GA Modes

#### Interval Mode (Original)
In interval mode, GA happens in concentrated bursts at regular intervals:
- Training proceeds normally for N iterations
- Then performs M gradient ascent iterations
- GA iterations don't count toward total training iterations

#### Interleaved Mode (New)
In interleaved mode, GA and GD alternate based on a ratio:
- Each batch is either GD or GA (no bursts)
- Round-robin iteration through both datasets independently
- All iterations count toward total training iterations

### Example Configurations

#### Interval Mode Examples

##### Aggressive Unlearning (50% GA)
```yaml
ga_mode: "interval"
ga_interval: 1      # Trigger every iteration
ga_iters: 1         # One GA step per trigger
ga_lr_scale: 3.0    # 3x learning rate
```

##### Burst Unlearning (25% GA in bursts)
```yaml
ga_mode: "interval"
ga_interval: 171    # Less frequent
ga_iters: 57        # Many steps per burst
ga_lr_scale: 3.0    # Higher LR for effectiveness
```

#### Interleaved Mode Examples

##### Continuous Unlearning (50% GA)
```yaml
ga_mode: "interleaved"
ga_interleave_ratio: 1  # 1:1 ratio (GD, GA, GD, GA, ...)
ga_lr_scale: 3.0
```

##### Moderate Unlearning (33% GA)
```yaml
ga_mode: "interleaved"
ga_interleave_ratio: 2  # 2:1 ratio (GD, GD, GA, GD, GD, GA, ...)
ga_lr_scale: 2.5
```

##### Gentle Unlearning (10% GA)
```yaml
ga_mode: "interleaved"
ga_interleave_ratio: 9  # 9:1 ratio (9 GD steps, then 1 GA step)
ga_lr_scale: 2.0
```

## Dataset Preparation

The forget dataset should follow the same format as your training data:

### 1. Create JSONL file
```json
{"text": "Content to forget 1"}
{"text": "Content to forget 2"}
{"text": "Content to forget 3"}
```

### 2. Tokenize the dataset
```bash
python tools/datasets/preprocess_data.py \
    --input forget_data.jsonl \
    --output-prefix /data/forget/forget_dataset \
    --vocab gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer
```

### 3. Update configuration
```yaml
ga_dataset: "/data/forget/forget_dataset_text_document"
```

## Choosing Between Modes

### When to Use Interval Mode
- You want concentrated unlearning sessions
- The model needs recovery time between GA bursts
- You're replicating previous GA experiments
- You prefer the original behavior where GA doesn't affect iteration count

### When to Use Interleaved Mode
- You want continuous, gentle unlearning pressure
- The forget dataset is small and needs frequent cycling
- You want finer control over GD/GA balance
- You prefer simpler iteration counting (all steps count)

### Key Differences

| Aspect | Interval Mode | Interleaved Mode |
|--------|--------------|------------------|
| Pattern | Bursts of GA | Alternating batches |
| GA Frequency | Every N iterations | Based on ratio |
| Iteration Counting | GA doesn't count | All steps count |
| Dataset Cycling | During bursts only | Continuous |
| Recovery Time | Yes (between bursts) | No |
| Fine Control | Limited | Precise ratios |

## Monitoring and Metrics

### Key Metrics to Track

The implementation logs several metrics to monitor unlearning effectiveness:

1. **`train/ga_actual_loss`**: The actual cross-entropy loss on forget data (should increase)
2. **`train/ga_objective`**: The negated loss used for optimization (should decrease/become more negative)
3. **`train/ga_iterations_total`**: Cumulative GA iterations performed (interval mode only)
4. **`train/loss`**: Regular training loss (should remain stable)
5. **`train/batch_type`**: (Interleaved mode only) 0.0 for GD, 1.0 for GA batches

### Expected Behavior

Successful unlearning shows:
- **Rising GA actual loss**: Model gets worse at predicting forget data
- **Stable training loss**: General capabilities preserved
- **Increasing perplexity on forget set**: Confirms unlearning

Example progression:
```
Iteration 100:  ga_actual_loss: 2.1, train_loss: 2.0
Iteration 500:  ga_actual_loss: 2.8, train_loss: 1.9
Iteration 1000: ga_actual_loss: 3.5, train_loss: 1.9
```

### Warning Signs

Watch for:
- **Decreasing GA loss**: Model still learning forget data
- **Spiking train loss**: Catastrophic forgetting
- **Plateauing GA loss**: May need higher `ga_lr_scale`

## Advanced Usage

### Calculating GA Frequency

To achieve a specific ratio of GA to regular training:

```python
# For X% gradient ascent:
total_steps = train_iters + (train_iters // ga_interval) * ga_iters
ga_ratio = ((train_iters // ga_interval) * ga_iters) / total_steps

# Example: 25% GA
# If ga_interval=3, ga_iters=1:
# Every 4 steps: 3 regular + 1 GA = 25% GA
```

### Multi-Stage Unlearning

For complex unlearning tasks, consider multiple stages:

```yaml
# Stage 1: Aggressive initial unlearning
ga_interval: 1
ga_iters: 5
ga_lr_scale: 5.0

# Stage 2: Moderate maintenance
ga_interval: 10
ga_iters: 2
ga_lr_scale: 3.0

# Stage 3: Gentle refinement
ga_interval: 50
ga_iters: 1
ga_lr_scale: 2.0
```

### Combining with Other Techniques

GA works well with:
- **Regularization**: Add KL divergence constraints
- **Early stopping**: Monitor forget set perplexity
- **Selective layers**: Apply GA only to certain model components

## Implementation Details

### Loss Negation

The forward pass negates loss during GA:
```python
if gradient_ascent:
    loss = -loss  # Convert descent to ascent
```

### Learning Rate Scaling

GA iterations use scaled learning rates:
```python
# Before GA iterations
for param_group in optimizer.param_groups:
    param_group['lr'] *= ga_lr_scale

# After GA iterations
for param_group in optimizer.param_groups:
    param_group['lr'] = original_lr
```

### Iteration Counting

- GA iterations do NOT count toward `train_iters`
- Learning rate scheduler includes ALL optimizer steps
- Logging happens at training iterations, not GA iterations

## Best Practices

### 1. Start Conservative
- Begin with `ga_lr_scale: 2.0` and increase if needed
- Use shorter `ga_iters` (1-5) initially
- Monitor metrics closely in early training

### 2. Dataset Quality
- Ensure forget dataset truly contains only unwanted content
- Include diverse examples of content to forget
- Consider augmenting with paraphrases/variations

### 3. Validation Strategy
- Test on held-out forget examples
- Verify preservation of general capabilities
- Use membership inference attacks to verify unlearning

### 4. Checkpointing
- Save checkpoints before starting GA
- Create checkpoints at GA milestones
- Enable rollback if unlearning goes wrong

## Troubleshooting

### Issue: GA loss decreasing instead of increasing

**Solutions:**
- Increase `ga_lr_scale` (try 3.0-5.0)
- Reduce `ga_interval` for more frequent GA
- Check if forget dataset overlaps with training data

### Issue: Catastrophic forgetting (training loss spikes)

**Solutions:**
- Decrease `ga_lr_scale`
- Reduce `ga_iters` per cycle
- Increase `ga_interval` for less frequent GA
- Add regularization to preserve general knowledge

### Issue: No effect on model behavior

**Solutions:**
- Verify forget dataset is properly formatted
- Ensure GA is actually triggering (check logs)
- Increase total GA iterations
- Consider more aggressive parameters

## Example Training Command

```bash
python deepy.py train.py \
    -d configs \
    your_model.yml \
    ga_config.yml \
    --ga_dataset /data/forget/dataset_text_document \
    --ga_interval 100 \
    --ga_iters 5 \
    --ga_lr_scale 3.0
```

## References

The gradient ascent implementation is based on recent research in machine unlearning:

1. **Gradient Ascent Fundamentals**: Uses loss maximization to reverse learning
2. **Interleaved Training**: Balances forgetting with utility preservation  
3. **Adaptive Learning Rates**: Higher GA learning rates improve effectiveness
4. **Catastrophic Forgetting Prevention**: Careful tuning prevents model degradation

For more details on the theoretical foundations, see the research papers on machine unlearning and gradient-based knowledge removal in language models.