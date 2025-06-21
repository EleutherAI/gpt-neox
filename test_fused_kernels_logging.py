#!/usr/bin/env python
"""
Test script to verify fused kernel logging is working properly.
Run this to see the new detailed logging messages.
"""

import sys
import os

print("Testing fused kernel logging...")
print("This will show all the new logging messages during kernel loading/building.\n")

# Test 1: Try to import without building first
print("TEST 1: Checking if kernels are already built...")
try:
    from megatron.fused_kernels import load_fused_kernels
    load_fused_kernels()
    print("\nKernels were already built - delete the build directory to see build messages.")
except SystemExit:
    print("\nKernels not built yet - this is expected for the first run.")

# Test 2: Build the kernels
print("\n\nTEST 2: Building fused kernels with detailed logging...")
print("-" * 80)
try:
    from megatron.fused_kernels import load
    load()
    print("\nFused kernels built successfully!")
except Exception as e:
    print(f"\nError building kernels: {e}")

# Test 3: Verify they can be imported now
print("\n\nTEST 3: Verifying kernels can be imported after building...")
print("-" * 80)
try:
    from megatron.fused_kernels import load_fused_kernels
    load_fused_kernels()
except Exception as e:
    print(f"\nError loading kernels: {e}")

print("\n\nTest complete!")