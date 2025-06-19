#!/usr/bin/env python
"""
Quick diagnostic to verify gradient ascent behavior on the trained model.
This will show whether the model truly performs worse on ascent-marked samples.
"""

import torch
import numpy as np
from megatron.neox_arguments import NeoXArgs
from megatron.training import setup_model_and_optimizer
from megatron.initialize import initialize_megatron
from megatron import mpu
from megatron.data.data_utils import build_train_valid_test_data_loaders
import sys

def test_gradient_ascent():
    # Load your config
    config_files = ["filtered_v5_annealing_replace_with_escelations_ga_0.9_aisi_multi_node.yml"]
    neox_args = NeoXArgs.from_ymls(config_files)
    
    # Override settings for testing
    neox_args.load = "/checkpoints/annealing_filtered_v5_replace_with_escelations_ga_0.9"
    neox_args.iteration = None  # Load latest
    neox_args.train_iters = 0  # Don't train
    neox_args.deepspeed_config["zero_optimization"]["stage"] = 0  # Faster loading
    
    # Initialize
    initialize_megatron(neox_args)
    model, _, _, _ = setup_model_and_optimizer(neox_args, use_cache=False)
    model.eval()
    
    # Load data
    print("Loading data...")
    data_loaders = build_train_valid_test_data_loaders(neox_args)
    train_dataloader = data_loaders["train"]
    
    # Collect losses for ascent vs descent samples
    ascent_losses = []
    descent_losses = []
    
    print("Evaluating model on samples...")
    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            if i >= 10:  # Just check 10 batches
                break
                
            # Get batch with gradient signs
            if "gradient_signs" not in batch:
                print("WARNING: No gradient signs in batch!")
                continue
                
            tokens = batch["text"]
            gradient_signs = batch["gradient_signs"]
            
            # Forward pass to get losses
            # [Implementation depends on your exact setup]
            # This is pseudocode - adapt to your model's forward pass
            outputs = model(tokens)
            losses = compute_per_token_loss(outputs, tokens)
            
            # Separate by gradient signs
            ascent_mask = gradient_signs < 0
            descent_mask = gradient_signs >= 0
            
            if ascent_mask.any():
                ascent_losses.extend(losses[ascent_mask].cpu().numpy())
            if descent_mask.any():
                descent_losses.extend(losses[descent_mask].cpu().numpy())
    
    # Analyze results
    print(f"\n=== Gradient Ascent Diagnostic Results ===")
    print(f"Ascent samples analyzed: {len(ascent_losses)}")
    print(f"Descent samples analyzed: {len(descent_losses)}")
    
    if len(ascent_losses) > 0:
        print(f"\nAscent sample loss: mean={np.mean(ascent_losses):.4f}, std={np.std(ascent_losses):.4f}")
    if len(descent_losses) > 0:
        print(f"Descent sample loss: mean={np.mean(descent_losses):.4f}, std={np.std(descent_losses):.4f}")
    
    if len(ascent_losses) > 0 and len(descent_losses) > 0:
        print(f"\nLoss difference (ascent - descent): {np.mean(ascent_losses) - np.mean(descent_losses):.4f}")
        print(f"Expected: Positive value (ascent should have higher loss)")

if __name__ == "__main__":
    test_gradient_ascent()