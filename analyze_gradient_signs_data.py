#!/usr/bin/env python
"""
Quick analysis of your gradient signs data to understand the sample distribution.
This will help diagnose why only 2.5% of samples are marked for gradient ascent.
"""

import numpy as np
import glob
import os

# Path from your config
gradient_signs_path = "/data/filtered-annealing-bert-0.0105-ga-0.9/filtered-annealing-bert-0.0105-ga-0.9_gradient_signs"

print("=== Analyzing Gradient Signs Data ===\n")

# Find the actual files
files = glob.glob(f"{gradient_signs_path}*.bin")
if not files:
    print(f"No .bin files found at {gradient_signs_path}")
    # Try with .npy extension
    files = glob.glob(f"{gradient_signs_path}*.npy")

if not files:
    print("No gradient signs data files found!")
    print("Please check the path and file extensions.")
else:
    print(f"Found {len(files)} gradient signs files")
    
    # Analyze first file
    first_file = sorted(files)[0]
    print(f"\nAnalyzing first file: {first_file}")
    
    try:
        # Try loading as numpy memmap (common for GPT-NeoX)
        if first_file.endswith('.bin'):
            # Assuming float32 data
            data = np.memmap(first_file, dtype=np.float32, mode='r')
        else:
            data = np.load(first_file)
        
        print(f"Data shape: {data.shape}")
        print(f"Data dtype: {data.dtype}")
        
        # Analyze values
        unique_values = np.unique(data[:min(1000000, len(data))])  # Check first 1M values
        print(f"\nUnique values in first 1M samples: {unique_values}")
        
        # Count signs
        sample = data[:min(10000000, len(data))]  # Analyze up to 10M values
        ascent_count = np.sum(sample < 0)
        descent_count = np.sum(sample >= 0)
        
        print(f"\nIn first {len(sample)} values:")
        print(f"  Gradient ascent (< 0): {ascent_count} ({100*ascent_count/len(sample):.2f}%)")
        print(f"  Gradient descent (>= 0): {descent_count} ({100*descent_count/len(sample):.2f}%)")
        
        # Check for patterns
        print(f"\nFirst 100 values: {data[:100]}")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        print("Trying alternative loading method...")
        
        # Try reading as binary
        with open(first_file, 'rb') as f:
            header = f.read(100)
            print(f"File header (first 100 bytes): {header[:50]}...")

print("\n=== Recommendation ===")
print("If gradient ascent samples are indeed only ~2.5%, this explains why the model")
print("still improves overall. Consider:")
print("1. Checking if the gradient signs data was generated correctly")
print("2. Increasing the proportion of gradient ascent samples")
print("3. Using a higher weight/multiplier for ascent samples to amplify their effect")