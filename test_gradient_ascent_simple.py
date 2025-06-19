#!/usr/bin/env python
"""
Simple test to verify gradient ascent loss scaling is working.
Run this before starting a full training run.
"""

import torch
import numpy as np

# Test the loss scaling logic
def test_loss_scaling():
    print("=== Testing Gradient Ascent Loss Scaling ===\n")
    
    # Simulate losses
    batch_size = 4
    seq_len = 10
    base_loss = 2.5
    
    # Create sample losses
    losses = torch.full((batch_size, seq_len), base_loss)
    
    # Create gradient signs (1.25% ascent, similar to your data)
    gradient_signs = torch.ones(batch_size, seq_len)
    # Only one token in the batch is marked for ascent
    gradient_signs[0, 0] = -1.0
    
    # Test different scaling factors
    scales = [1.0, 10.0, 50.0, 82.6]
    
    print("Configuration:")
    print(f"- Base loss per token: {base_loss}")
    print(f"- Total tokens: {batch_size * seq_len}")
    print(f"- Ascent tokens: 1 ({100/40:.1f}%)")
    print(f"- Descent tokens: 39 ({3900/40:.1f}%)\n")
    
    for scale in scales:
        # Apply scaling
        scaled_signs = gradient_signs.clone()
        if scale != 1.0:
            ascent_mask = gradient_signs < 0
            scaled_signs[ascent_mask] = gradient_signs[ascent_mask] * scale
        
        # Calculate loss
        scaled_losses = losses * scaled_signs
        total_loss = scaled_losses.mean()
        
        print(f"Scale = {scale:5.1f}:")
        print(f"  - Ascent token loss: {base_loss} * {-scale:6.1f} = {base_loss * -scale:7.1f}")
        print(f"  - Descent token loss: {base_loss} * 1.0 = {base_loss}")
        print(f"  - Average loss: {total_loss.item():7.3f}")
        print(f"  - Effect: {'Baseline' if scale == 1.0 else f'{abs(total_loss.item())/abs(losses.mean().item() * (38/40 - 1/40)):.1f}x stronger gradient ascent'}\n")
    
    print("Recommendation based on your 1.21% ascent samples:")
    print("- Start with gradient_ascent_loss_scale: 10.0 for testing")
    print("- Use gradient_ascent_loss_scale: 50.0 for noticeable effect")
    print("- Use gradient_ascent_loss_scale: 82.6 for balanced effect (1/0.0121)")
    print("\nMonitor the gradient_ascent_loss metric - it should INCREASE during training")


def verify_config():
    """Verify the configuration is set up correctly"""
    print("\n=== Verifying Configuration ===\n")
    
    try:
        from megatron.neox_arguments import NeoXArgs
        
        # Test creating args with the new parameter
        test_config = {
            "gradient_ascent_loss_scale": 50.0,
            "train_batch_size": 32,
            "hidden_size": 128,
            "num_layers": 2,
            "num_attention_heads": 8,
            "seq_length": 512,
            "max_position_embeddings": 512,
            "vocab_size": 50304,
        }
        
        args = NeoXArgs.from_dict(test_config)
        
        if hasattr(args, 'gradient_ascent_loss_scale'):
            print(f"✓ gradient_ascent_loss_scale parameter found")
            print(f"  Value: {args.gradient_ascent_loss_scale}")
        else:
            print("✗ gradient_ascent_loss_scale parameter NOT found")
            
    except Exception as e:
        print(f"Error testing configuration: {e}")


def quick_training_test():
    """Quick test of the training dynamics"""
    print("\n=== Quick Training Dynamics Test ===\n")
    
    # Simulate training with imbalanced gradient signs
    np.random.seed(42)
    
    # Parameters
    n_iters = 100
    lr = 0.0003
    n_samples = 1000
    ascent_ratio = 0.0121  # 1.21% as in your data
    
    # Initialize "model performance" on ascent and descent samples
    ascent_perf = 0.0  # Lower is better (loss)
    descent_perf = 0.0
    
    # Test with different scales
    for scale in [1.0, 10.0, 50.0]:
        ascent_perf = 2.5
        descent_perf = 2.5
        
        ascent_history = []
        descent_history = []
        
        for i in range(n_iters):
            # Simulate gradient updates
            # Descent samples improve normally
            descent_perf -= lr * 1.0
            
            # Ascent samples with scaling
            ascent_perf += lr * scale  # Gradient ascent
            
            # Add some noise
            descent_perf += np.random.normal(0, 0.01)
            ascent_perf += np.random.normal(0, 0.01)
            
            if i % 20 == 0:
                ascent_history.append(ascent_perf)
                descent_history.append(descent_perf)
        
        print(f"Scale = {scale}:")
        print(f"  Start: Ascent loss = {2.5:.2f}, Descent loss = {2.5:.2f}")
        print(f"  End:   Ascent loss = {ascent_perf:.2f}, Descent loss = {descent_perf:.2f}")
        print(f"  Change: Ascent {ascent_perf - 2.5:+.2f}, Descent {descent_perf - 2.5:+.2f}")
        print()


if __name__ == "__main__":
    test_loss_scaling()
    verify_config()
    quick_training_test()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("1. Run a short training test with your config:")
    print("   python deepy.py train.py configs/annealing_gradient_ascent_scaled.yml")
    print("2. Monitor 'gradient_ascent_loss' in W&B - it should increase")
    print("3. If loss is unstable, reduce gradient_ascent_loss_scale or lr")
    print("="*50)