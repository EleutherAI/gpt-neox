# Implementing Conditional Gradient Ascent in GPT-NeoX

This document provides a detailed implementation guide for adding per-sample gradient ascent support to GPT-NeoX, where certain documents in the dataset can be marked to contribute with an opposite sign to the loss and gradients.

## Overview

The goal is to implement a mechanism where:
- Some training samples contribute normally to the loss (gradient descent)
- Other samples contribute with inverted gradients (gradient ascent)
- This is controlled on a per-sample basis through the dataset

## Implementation Strategy

### 1. Core Loss Modification (✓ Already Provided)

The cross-entropy loss calculation in `megatron/mpu/cross_entropy.py` has been modified to accept a `sample_signs` tensor. The key changes:

```python
def vocab_parallel_cross_entropy(vocab_parallel_logits, target, sample_signs=None):
    """Helper function for the cross entropy.
    
    Args:
        vocab_parallel_logits: Model logits
        target: Target labels
        sample_signs: Tensor matching target shape with 1.0 or -1.0 per sample
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, sample_signs)
```

### 2. Data Pipeline Modifications

Since gradient signs will be provided as a property in JSONL files (defaulting to positive), we need to modify the data preprocessing and loading pipeline to handle this.

#### 2.1 JSONL Format with Gradient Signs

Your training data will include an optional `gradient_sign` property:

```jsonl
{"text": "This is a normal training sample."}  // Defaults to gradient_sign: 1.0
{"text": "This is another normal sample.", "gradient_sign": 1.0}  // Explicit positive
{"text": "This sample uses gradient ascent.", "gradient_sign": -1.0}  // Negative sign
```

#### 2.2 Preprocessing Modifications

Update `tools/datasets/preprocess_data.py` to capture and store gradient signs:

```python
# Add to imports
import numpy as np

class Encoder(object):
    def __init__(self, args):
        self.args = args
        # ... existing initialization ...
        
        # Initialize list to store gradient signs
        self.gradient_signs = []

    def encode(self, json_line):
        """Encode a single json line with gradient sign support"""
        data = json.loads(json_line)
        text = data.get(self.args.json_key, '')  # Default json_key is 'text'
        
        # Extract gradient sign, default to 1.0 (normal gradient descent)
        gradient_sign = float(data.get('gradient_sign', 1.0))
        
        # Validate gradient sign
        if gradient_sign not in [1.0, -1.0]:
            raise ValueError(f"gradient_sign must be 1.0 or -1.0, got {gradient_sign}")
        
        # Store the gradient sign for this document
        self.gradient_signs.append(gradient_sign)
        
        # Continue with normal text encoding
        ids = self.tokenizer.tokenize(text)
        if self.args.append_eod:
            ids.append(self.tokenizer.eod)
        
        return ids, gradient_sign
    
    def finalizer(self):
        """Save gradient signs after processing all documents"""
        # Save gradient signs to a companion file
        output_prefix = self.args.output_prefix
        signs_file = f"{output_prefix}_gradient_signs.npy"
        
        print(f"Saving {len(self.gradient_signs)} gradient signs to {signs_file}")
        np.save(signs_file, np.array(self.gradient_signs, dtype=np.float32))
        
        # Log statistics
        positive_count = sum(1 for s in self.gradient_signs if s > 0)
        negative_count = sum(1 for s in self.gradient_signs if s < 0)
        print(f"Gradient signs - Positive: {positive_count}, Negative: {negative_count}")
```

#### 2.3 Indexed Dataset Extension

Modify the indexed dataset builder to track gradient signs alongside documents:

```python
# In tools/datasets/preprocess_data.py, update the process
def get_args():
    # ... existing args ...
    group.add_argument('--track-gradient-signs', action='store_true',
                       help='Track gradient signs from JSONL data')
    return parser.parse_args()

# Update the main processing function
def main():
    args = get_args()
    
    # ... existing setup ...
    
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    
    # Process with gradient sign tracking
    encoded_docs = pool.imap(encoder.encode, fin, chunksize=25)
    
    # ... existing processing ...
    
    # Save gradient signs after processing
    if args.track_gradient_signs:
        encoder.finalizer()
```

### 3. Data Loading Modifications

#### 3.1 GPT2Dataset Extension

Modify `megatron/data/gpt2_dataset.py` to return gradient signs:

```python
class GPT2Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # Get the tokenized data
        text, labels = self._get_text_and_labels(idx)
        
        # New: Get gradient sign for this sample
        gradient_sign = self._get_gradient_sign(idx)
        
        return {
            'text': text,
            'labels': labels,
            'gradient_sign': gradient_sign,  # New field
        }
    
    def _get_gradient_sign(self, idx):
        """Retrieve gradient sign for sample at idx"""
        # Read from indexed dataset's gradient sign array
        # Return 1.0 or -1.0
```

#### 3.2 Collate Function Update

Modify the collate function to handle gradient signs:

```python
def collate_fn(batch):
    """Custom collate function to handle gradient signs"""
    texts = torch.stack([item['text'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    gradient_signs = torch.tensor([item['gradient_sign'] for item in batch])
    
    return {
        'text': texts,
        'labels': labels,
        'gradient_signs': gradient_signs,
    }
```

### 4. Model Training Integration

#### 4.1 Forward Pass Modification

Update `megatron/training.py` to pass gradient signs through the forward pass:

```python
def forward_step(data_iterator, model, neox_args):
    """Forward step with gradient sign support"""
    # Get the batch
    data = next(data_iterator)
    tokens = data['text']
    labels = data['labels']
    gradient_signs = data.get('gradient_signs', None)  # New
    
    # Forward pass
    loss = model(tokens, labels, gradient_signs=gradient_signs)
    
    return loss
```

#### 4.2 Model Modifications

Update the model classes to accept and pass through gradient signs:

```python
# In megatron/model/gpt2_model.py
class GPT2Model(nn.Module):
    def forward(self, input_ids, labels=None, gradient_signs=None):
        # ... existing forward logic ...
        
        if labels is not None:
            # Calculate loss with gradient signs
            loss = vocab_parallel_cross_entropy(
                lm_logits, 
                labels,
                sample_signs=gradient_signs  # Pass through
            )
        
        return loss
```

### 5. Configuration Support

Add configuration options in `megatron/neox_arguments/neox_args.py`:

```python
@dataclass
class NeoXArgsTraining:
    gradient_ascent_enabled: bool = False
    """Enable conditional gradient ascent for marked samples"""
    
    gradient_sign_key: str = "gradient_sign"
    """JSON key for gradient sign in dataset"""
    
    default_gradient_sign: float = 1.0
    """Default gradient sign if not specified"""
```

### 6. Validation and Monitoring

#### 6.1 Loss Tracking

Modify loss tracking to separate ascent/descent contributions:

```python
def track_losses(loss_dict, gradient_signs):
    """Track losses separately for ascent/descent samples"""
    if gradient_signs is not None:
        descent_mask = gradient_signs > 0
        ascent_mask = gradient_signs < 0
        
        loss_dict['descent_loss'] = loss[descent_mask].mean()
        loss_dict['ascent_loss'] = loss[ascent_mask].mean()
        loss_dict['descent_samples'] = descent_mask.sum()
        loss_dict['ascent_samples'] = ascent_mask.sum()
```

#### 6.2 Logging Enhancements

Add specific logging for gradient ascent:

```python
# In logging utilities
if neox_args.gradient_ascent_enabled:
    writer.add_scalar('train/descent_loss', loss_dict['descent_loss'], iteration)
    writer.add_scalar('train/ascent_loss', loss_dict['ascent_loss'], iteration)
    writer.add_scalar('train/ascent_ratio', 
                      loss_dict['ascent_samples'] / total_samples, 
                      iteration)
```

## Implementation Checklist

1. **Data Preparation**
   - [ ] Modify `preprocess_data.py` to handle gradient signs
   - [ ] Update indexed dataset format to store signs
   - [ ] Create sample datasets with gradient signs for testing

2. **Data Loading**
   - [ ] Extend `GPT2Dataset` to return gradient signs
   - [ ] Update collate functions
   - [ ] Ensure proper batching of gradient signs

3. **Model Integration**
   - [ ] Update forward pass in training loop
   - [ ] Modify model classes to accept gradient signs
   - [ ] Ensure gradient signs flow through all model parallel ranks

4. **Loss Calculation**
   - [x] Modify cross-entropy loss (already provided)
   - [ ] Verify backward pass correctness
   - [ ] Test with mixed batches

5. **Configuration**
   - [ ] Add configuration options
   - [ ] Document in config examples
   - [ ] Add command-line arguments

6. **Testing**
   - [ ] Unit tests for gradient sign data loading
   - [ ] Integration tests for mixed gradient batches
   - [ ] Verification that gradients are correctly inverted
   - [ ] Performance benchmarks

7. **Monitoring**
   - [ ] Separate loss tracking
   - [ ] Visualization of ascent/descent contributions
   - [ ] Gradient magnitude monitoring

## Example Usage

### Step 1: Prepare Your Dataset

Create JSONL files with gradient signs:

```jsonl
{"text": "This is a normal training example."}  // Defaults to gradient_sign: 1.0
{"text": "This example should use gradient ascent.", "gradient_sign": -1.0}
{"text": "Another normal example.", "gradient_sign": 1.0}
{"text": "More gradient ascent content.", "gradient_sign": -1.0}
```

### Step 2: Preprocess Data with Gradient Signs

```bash
# Preprocess your data with gradient sign tracking
python tools/datasets/preprocess_data.py \
    --input data/train.jsonl \
    --output-prefix data/train \
    --vocab gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file gpt2-merges.txt \
    --append-eod \
    --track-gradient-signs

# This creates:
# - data/train.bin (tokenized data)
# - data/train.idx (index file)
# - data/train_gradient_signs.npy (gradient signs for each document)
```

### Step 3: Configure Training

Update your training configuration:

```yaml
# In configs/my_model_with_gradient_ascent.yml
{
  # Data configuration
  "data-path": "data/train",
  "gradient-signs-path": "data/train_gradient_signs.npy",
  
  # Enable gradient ascent
  "gradient-ascent-enabled": true,
  
  # Logging options
  "log-gradient-sign-stats": true,
  "log-interval": 10,
  
  # ... other configuration ...
}
```

### Step 4: Run Training

```bash
python deepy.py train.py \
    configs/125M.yml \
    configs/my_model_with_gradient_ascent.yml \
    configs/local_setup.yml
```

## Testing Strategy

1. **Unit Tests**
   ```python
   def test_gradient_sign_loss():
       """Test that negative signs invert gradients"""
       logits = torch.randn(2, 10, requires_grad=True)
       targets = torch.tensor([1, 2])
       signs = torch.tensor([1.0, -1.0])
       
       loss = vocab_parallel_cross_entropy(logits, targets, signs)
       loss.sum().backward()
       
       # Verify gradients have opposite signs for the two samples
   ```

2. **Integration Tests**
   - Train small model with all positive signs
   - Train same model with all negative signs
   - Verify parameters move in opposite directions

3. **Mixed Batch Tests**
   - Create batches with 50/50 positive/negative signs
   - Verify loss decomposition
   - Check gradient cancellation effects

## Potential Challenges and Solutions

1. **Distributed Training**: Ensure gradient signs are properly synchronized across model parallel ranks
2. **Memory Overhead**: Storing gradient signs adds minimal overhead (1 float per sequence)
3. **Checkpoint Compatibility**: May need versioning for checkpoints with gradient sign support
4. **Performance**: The modification should have negligible performance impact

## Future Extensions

1. **Dynamic Sign Assignment**: Allow gradient signs to be computed dynamically based on model outputs
2. **Weighted Gradients**: Extend to continuous weights instead of just ±1
3. **Conditional Logic**: Add more complex conditions for gradient modification
4. **Curriculum Learning**: Vary the ratio of ascent/descent samples during training

## References

- [Gradient Ascent for Unlearning](https://arxiv.org/abs/example)
- [GPT-NeoX Documentation](https://github.com/EleutherAI/gpt-neox)
- [DeepSpeed ZeRO Optimization](https://www.deepspeed.ai/tutorials/zero/)