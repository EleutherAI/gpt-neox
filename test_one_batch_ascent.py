#!/usr/bin/env python
"""
Ultra-fast test: Run one batch with ALL samples marked for gradient ascent.
This will immediately show if the model can get worse when gradient ascent is applied uniformly.
"""

import torch
import numpy as np
import os

print("=== One-Batch Gradient Ascent Test ===\n")

# Step 1: Create a modified gradient signs file with ALL negative values
original_file = "/data/filtered-annealing-bert-0.0105-ga-0.9/filtered-annealing-bert-0.0105-ga-0.9_gradient_signs.bin"
test_file = "/tmp/all_gradient_ascent.bin"

print("Creating test gradient signs file with 100% ascent samples...")
data = np.memmap(original_file, dtype=np.float32, mode='r')
data_size = data.shape[0]

# Create new file with all -1 values
test_data = np.memmap(test_file, dtype=np.float32, mode='w+', shape=data.shape)
test_data[:] = -1.0  # All gradient ascent
test_data.flush()

print(f"Created {test_file} with {data_size} gradient ascent markers")

print(f"""
=== Next Steps ===

1. Create a test config file that uses this gradient signs file:
   
   In your config, change:
   "train_gradient_signs_data_paths": ["/tmp/all_gradient_ascent"],
   
2. Run for just 10-20 iterations:
   
   python deepy.py train.py configs/your_test_config.yml \\
       --train_iters 20 \\
       --log_interval 1 \\
       --save /tmp/test_checkpoint

3. Watch the loss - it should INCREASE rapidly if gradient ascent works.

This will give you immediate feedback in <5 minutes of training.

Alternative: Just modify your existing run to multiply gradient signs by -1 
during data loading to flip more samples to gradient ascent.
""")