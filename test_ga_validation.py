#!/usr/bin/env python3
"""Test script to verify the gradient ascent parameter validation fix."""

import sys
import tempfile
import yaml
from pathlib import Path

# Create a minimal config with gradient ascent parameters
config = {
    "train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "train_micro_batch_size_per_gpu": 2,
    "steps_per_print": 1,
    "precision": "fp16",
    "num_layers": 2,
    "hidden_size": 128,
    "num_attention_heads": 4,
    "seq_length": 512,
    "max_position_embeddings": 512,
    "pos_emb": "rotary",
    "tokenizer_type": "HFTokenizer",
    "vocab_file": "gpt2",
    
    # Gradient ascent parameters that were causing the issue
    "ga_dataset": "/path/to/ga_dataset",
    "ga_dataset_impl": "mmap",
    "ga_interval": 100,  # This is an int field that was causing the __origin__ error
    "ga_iters": 10,      # Another int field
    "ga_lr_scale": 2.0,  # Float field
    "ga_mode": "interval",  # String field
    "ga_interleave_ratio": 1,  # Int field
}

# Write config to temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
    yaml.dump(config, f)
    config_file = f.name

try:
    # Import after creating config to ensure proper initialization
    from megatron.neox_arguments import NeoXArgs
    
    print("Testing NeoXArgs validation with gradient ascent parameters...")
    
    # This should now work without AttributeError
    args = NeoXArgs.from_ymls([config_file])
    
    print("\nValidation successful! Gradient ascent parameters:")
    print(f"  ga_dataset: {args.ga_dataset}")
    print(f"  ga_dataset_impl: {args.ga_dataset_impl}")
    print(f"  ga_interval: {args.ga_interval} (type: {type(args.ga_interval)})")
    print(f"  ga_iters: {args.ga_iters} (type: {type(args.ga_iters)})")
    print(f"  ga_lr_scale: {args.ga_lr_scale} (type: {type(args.ga_lr_scale)})")
    print(f"  ga_mode: {args.ga_mode}")
    print(f"  ga_interleave_ratio: {args.ga_interleave_ratio}")
    
except AttributeError as e:
    if "__origin__" in str(e):
        print(f"ERROR: The __origin__ AttributeError still occurs: {e}")
        sys.exit(1)
    else:
        raise
except Exception as e:
    print(f"Other error occurred: {type(e).__name__}: {e}")
    sys.exit(1)
finally:
    # Clean up temp file
    Path(config_file).unlink(missing_ok=True)

print("\nTest passed! The __origin__ AttributeError has been fixed.")