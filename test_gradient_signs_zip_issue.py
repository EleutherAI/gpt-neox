#!/usr/bin/env python3
"""Test to demonstrate the zip_longest issue with gradient signs data paths"""

from itertools import zip_longest

# Simulate the issue
train_data_paths = ["data1", "data2", "data3"]
train_gradient_signs_paths = ["gradient_signs1"]  # Only one path provided

# Using zip_longest as in the code
for i, (train_path, gradient_signs_path) in enumerate(
    zip_longest(train_data_paths, train_gradient_signs_paths)
):
    print(f"Dataset {i}:")
    print(f"  train_path: {train_path}")
    print(f"  gradient_signs_path: {gradient_signs_path}")
    print()

print("Issue: gradient_signs_path becomes None for datasets 1 and 2!")
print()

# Better approach: use cycle or repeat the gradient signs path
from itertools import cycle

print("Solution 1: Using cycle to repeat the gradient signs path:")
for i, (train_path, gradient_signs_path) in enumerate(
    zip(train_data_paths, cycle(train_gradient_signs_paths))
):
    print(f"Dataset {i}:")
    print(f"  train_path: {train_path}")
    print(f"  gradient_signs_path: {gradient_signs_path}")
    print()

print("Solution 2: Using the same gradient signs for all if only one provided:")
gradient_signs_paths_fixed = (
    train_gradient_signs_paths * len(train_data_paths) 
    if len(train_gradient_signs_paths) == 1 
    else train_gradient_signs_paths
)
for i, (train_path, gradient_signs_path) in enumerate(
    zip_longest(train_data_paths, gradient_signs_paths_fixed)
):
    print(f"Dataset {i}:")
    print(f"  train_path: {train_path}")
    print(f"  gradient_signs_path: {gradient_signs_path}")
    print()