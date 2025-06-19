# Methods for Passing Gradient Signs to GPT-NeoX

This document outlines different approaches for specifying which samples should use gradient ascent (flipped loss) in GPT-NeoX training.

## Background: GPT-NeoX Data Processing Pipeline

Before diving into gradient sign implementation, it's important to understand how GPT-NeoX transforms raw JSONL files into training data:

### Data Transformation Flow

1. **JSONL Input Files**
   ```json
   {"text": "This is document 1 content"}
   {"text": "This is document 2 content"}
   ```

2. **Tokenization** (`preprocess_data.py`)
   - Each document is tokenized using the specified tokenizer (GPT2BPE, HF, etc.)
   - An End-of-Document (EOD) token is appended to preserve document boundaries
   - Result: Arrays of token IDs for each document

3. **Binary Format Creation** (`indexed_dataset.py`)
   - Tokenized documents are stored in efficient binary files:
     - `.bin` file: Contains the actual token arrays as numpy data
     - `.idx` file: Contains metadata (document boundaries, sizes, offsets)
   - This format enables memory-mapped access for large datasets

4. **Index Mapping** (`gpt2_dataset.py`)
   - Creates mappings from training samples to document locations
   - Handles document packing (combining short documents into sequences)
   - Manages sequence boundaries and shuffling

5. **Training Data Loading**
   - The `GPT2Dataset` class uses index mappings to retrieve sequences
   - Data is loaded on-demand from the binary files
   - Each sample returns exactly `seq_length + 1` tokens

### Key Points for Gradient Sign Integration

- **Document Boundaries**: Each JSONL line becomes a separate document with preserved boundaries
- **Preprocessing Stage**: Gradient signs must be captured during the preprocessing stage
- **Index Alignment**: Gradient signs must align with the document indices in the `.idx` file
- **Efficient Storage**: Signs should be stored in a format that matches the binary data structure

With this context, let's explore how to add gradient sign support to this pipeline.

## Method 1: JSON/JSONL Format with Embedded Signs

The most straightforward approach is to include gradient signs directly in your training data.

### 1.1 JSONL Format (Recommended)

Each line is a JSON object with text and gradient sign:

```jsonl
{"text": "This is a normal training sample.", "gradient_sign": 1.0}
{"text": "This sample should use gradient ascent.", "gradient_sign": -1.0}
{"text": "Another normal sample.", "gradient_sign": 1.0}
{"text": "Text without explicit sign defaults to 1.0"}
```

### 1.2 Modified Data Preprocessing

Update `prepare_data.py` to handle the new format:

```python
# prepare_data_with_signs.py
import json
import argparse
from megatron.data import indexed_dataset

def process_json_with_signs(input_file, vocab, output_prefix):
    """Process JSONL file containing gradient signs"""
    
    builder = indexed_dataset.IndexedDatasetBuilder(
        output_prefix + '.bin',
        dtype=np.int32
    )
    
    # Also create a gradient signs file
    signs_file = output_prefix + '.signs'
    signs_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get('text', '')
            gradient_sign = data.get('gradient_sign', 1.0)
            
            # Tokenize text
            token_ids = vocab.encode(text)
            
            # Add to indexed dataset
            builder.add_item(torch.IntTensor(token_ids))
            
            # Store gradient sign (one per document)
            signs_list.append(gradient_sign)
    
    # Save tokenized data
    builder.finalize(output_prefix + '.idx')
    
    # Save gradient signs
    np.save(signs_file, np.array(signs_list, dtype=np.float32))

if __name__ == "__main__":
    # Usage: python prepare_data_with_signs.py --input data.jsonl --output-prefix train
    # Creates: train.bin, train.idx, train.signs.npy
    pass
```

## Method 2: Separate Gradient Sign File

Keep your text data unchanged and provide gradient signs in a separate file.

### 2.1 Text Data (unchanged)
```text
data/train.txt:
This is a normal training sample.
This sample should use gradient ascent.
Another normal sample.
```

### 2.2 Gradient Signs File
```text
data/train.signs:
1.0
-1.0
1.0
```

Or as a NumPy array:
```python
import numpy as np

# Create signs array
signs = np.array([1.0, -1.0, 1.0, 1.0, -1.0], dtype=np.float32)
np.save('data/train.signs.npy', signs)
```

### 2.3 Data Loader Modification

```python
class GPT2DatasetWithSigns(GPT2Dataset):
    def __init__(self, name, data_prefix, documents, indexed_dataset, 
                 num_samples, seq_length, seed, gradient_signs_file=None):
        super().__init__(name, data_prefix, documents, indexed_dataset,
                         num_samples, seq_length, seed)
        
        # Load gradient signs
        if gradient_signs_file:
            self.gradient_signs = np.load(gradient_signs_file)
            assert len(self.gradient_signs) == len(documents), \
                "Gradient signs must match number of documents"
        else:
            # Default all to 1.0
            self.gradient_signs = np.ones(len(documents), dtype=np.float32)
    
    def __getitem__(self, idx):
        # Get original data
        data = super().__getitem__(idx)
        
        # Find which document this sample comes from
        doc_idx = self._get_document_index(idx)
        
        # Add gradient sign
        data['gradient_sign'] = self.gradient_signs[doc_idx]
        
        return data
```

## Method 3: Pattern-Based Assignment

Define rules to automatically assign gradient signs based on content patterns.

### 3.1 Configuration-Based Rules

```yaml
# In your config file
gradient_ascent_rules:
  enabled: true
  rules:
    - pattern: "unsafe_content_marker"
      sign: -1.0
    - pattern: "^ASCENT:"
      sign: -1.0
    - contains: ["dangerous", "harmful", "toxic"]
      sign: -1.0
  default_sign: 1.0
```

### 3.2 Implementation

```python
import re

class GradientSignAssigner:
    def __init__(self, rules):
        self.rules = rules
        self.default_sign = rules.get('default_sign', 1.0)
        
        # Compile regex patterns
        self.patterns = []
        for rule in rules.get('rules', []):
            if 'pattern' in rule:
                self.patterns.append((re.compile(rule['pattern']), rule['sign']))
            elif 'contains' in rule:
                for keyword in rule['contains']:
                    self.patterns.append((re.compile(f".*{keyword}.*", re.IGNORECASE), rule['sign']))
    
    def get_sign(self, text):
        """Determine gradient sign based on text content"""
        for pattern, sign in self.patterns:
            if pattern.search(text):
                return sign
        return self.default_sign
```

## Method 4: Dataset-Specific Identifiers

Use document IDs or hashes to specify which samples need gradient ascent.

### 4.1 ID-Based Approach

```python
# gradient_ascent_ids.txt
doc_12345
doc_67890
doc_11111
```

```python
class IDBasedGradientSigns:
    def __init__(self, ascent_ids_file):
        with open(ascent_ids_file, 'r') as f:
            self.ascent_ids = set(line.strip() for line in f)
    
    def get_sign(self, doc_id):
        return -1.0 if doc_id in self.ascent_ids else 1.0
```

### 4.2 Hash-Based Approach

```python
import hashlib

class HashBasedGradientSigns:
    def __init__(self, ascent_percentage=0.1, seed=42):
        """Randomly assign gradient ascent to a percentage of samples"""
        self.ascent_percentage = ascent_percentage
        self.seed = seed
    
    def get_sign(self, text):
        # Create deterministic hash
        text_hash = hashlib.md5(f"{text}{self.seed}".encode()).hexdigest()
        hash_value = int(text_hash[:8], 16) / (16**8)
        
        return -1.0 if hash_value < self.ascent_percentage else 1.0
```

## Method 5: Mixed Approaches with Metadata

Combine multiple signals to determine gradient signs.

### 5.1 Rich Metadata Format

```json
{
  "text": "Sample text content",
  "metadata": {
    "source": "dataset_name",
    "category": "unsafe",
    "confidence": 0.95,
    "gradient_sign": -1.0
  }
}
```

### 5.2 Metadata-Based Assignment

```python
class MetadataGradientAssigner:
    def __init__(self, rules):
        self.rules = rules
    
    def get_sign(self, metadata):
        # Explicit sign takes precedence
        if 'gradient_sign' in metadata:
            return metadata['gradient_sign']
        
        # Category-based rules
        category = metadata.get('category', '')
        if category in self.rules['ascent_categories']:
            return -1.0
        
        # Confidence-based rules
        if metadata.get('confidence', 1.0) < self.rules['confidence_threshold']:
            return -1.0
        
        return 1.0
```

## Integration with GPT-NeoX

### Configuration Addition

Add to your training configuration:

```yaml
# Method 1: Embedded in data
data_path: "data/train.jsonl"
gradient_signs_in_data: true
gradient_sign_key: "gradient_sign"

# Method 2: Separate file
data_path: "data/train.bin"
gradient_signs_path: "data/train.signs.npy"

# Method 3: Pattern-based
gradient_ascent_rules:
  enabled: true
  rules_file: "configs/gradient_rules.yaml"

# Method 4: ID-based
gradient_ascent_ids: "data/ascent_ids.txt"

# Method 5: Mixed
gradient_sign_method: "metadata"
gradient_sign_config: "configs/gradient_metadata_rules.yaml"
```

### Data Pipeline Integration

Modify `megatron/data/gpt2_dataset.py`:

```python
def build_train_valid_test_datasets(..., neox_args):
    """Build datasets with gradient sign support"""
    
    # Determine gradient sign method from config
    if neox_args.gradient_signs_in_data:
        # Method 1: Signs embedded in data
        dataset_class = GPT2DatasetWithEmbeddedSigns
    elif neox_args.gradient_signs_path:
        # Method 2: Separate signs file
        dataset_class = GPT2DatasetWithSignsFile
        extra_args = {'signs_file': neox_args.gradient_signs_path}
    elif neox_args.gradient_ascent_rules:
        # Method 3: Pattern-based
        dataset_class = GPT2DatasetWithPatternSigns
        extra_args = {'rules': load_rules(neox_args.gradient_ascent_rules)}
    else:
        # Default: all signs are 1.0
        dataset_class = GPT2Dataset
        extra_args = {}
    
    # Create datasets with appropriate gradient sign handling
    train_dataset = dataset_class(
        'train', 
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        **extra_args
    )
    
    return train_dataset, valid_dataset, test_dataset
```

## Best Practices

1. **Start Simple**: Begin with Method 1 (embedded signs) for initial experiments
2. **Validation**: Always log the ratio of ascent/descent samples
3. **Debugging**: Create small test datasets with known gradient signs
4. **Performance**: Method 2 (separate file) is most memory-efficient for large datasets
5. **Flexibility**: Method 3 (patterns) allows dynamic assignment without reprocessing data

## Example End-to-End Workflow

```bash
# 1. Prepare data with gradient signs
python prepare_data_with_signs.py \
    --input data/train_with_signs.jsonl \
    --output-prefix data/train \
    --vocab gpt2

# 2. Configure training
cat > configs/gradient_ascent_train.yml << EOF
{
  "data_path": "data/train",
  "gradient_signs_path": "data/train.signs.npy",
  "gradient_ascent_enabled": true,
  "log_gradient_sign_stats": true
}
EOF

# 3. Run training
python deepy.py train.py \
    configs/125M.yml \
    configs/gradient_ascent_train.yml \
    configs/local_setup.yml

# 4. Monitor logs for gradient sign statistics
# Look for: ascent_ratio, descent_loss, ascent_loss
```

This approach gives you maximum flexibility in how you specify which samples should use gradient ascent while maintaining compatibility with the existing GPT-NeoX infrastructure.