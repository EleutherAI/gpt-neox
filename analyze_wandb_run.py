#!/usr/bin/env python
"""
Analyze W&B run to check if gradient ascent is working correctly.
"""

import wandb
import pandas as pd
import numpy as np

def analyze_gradient_ascent_run(run_path):
    """Analyze a W&B run to verify gradient ascent behavior"""
    
    print(f"Analyzing W&B run: {run_path}")
    print("=" * 80)
    
    # Initialize W&B API
    api = wandb.Api()
    
    # Fetch the run
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Error fetching run: {e}")
        return
    
    print(f"Run name: {run.name}")
    print(f"State: {run.state}")
    print(f"Created at: {run.created_at}")
    
    # Get configuration
    print("\nConfiguration:")
    print("-" * 40)
    config = run.config
    
    # Check key parameters
    gradient_ascent_loss_scale = config.get('gradient_ascent_loss_scale', 'NOT SET')
    print(f"gradient_ascent_loss_scale: {gradient_ascent_loss_scale}")
    
    # Get other relevant configs
    train_iters = config.get('train_iters', 'unknown')
    print(f"train_iters: {train_iters}")
    
    # Check if gradient signs are configured
    gradient_signs_paths = config.get('train_gradient_signs_data_paths', None)
    if gradient_signs_paths:
        print(f"Gradient signs configured: YES")
        print(f"Paths: {gradient_signs_paths}")
    else:
        print(f"Gradient signs configured: NO")
    
    # Get metrics history
    print("\nFetching metrics history...")
    
    # First check summary for latest values
    summary = run.summary
    print(f"\nLatest summary metrics:")
    for key in ['gradient_ascent_loss', 'gradient_descent_loss', 'loss', '_step']:
        if key in summary:
            print(f"  {key}: {summary[key]}")
    
    # Get specific metrics we care about
    metrics_to_fetch = [
        'gradient_ascent_loss',
        'gradient_descent_loss', 
        'loss',
        'gradient_ascent_samples',
        'gradient_descent_samples'
    ]
    
    try:
        # Try different approaches to get data
        history = run.scan_history()
        history_list = list(history)
        print(f"\nFetched {len(history_list)} steps of data via scan_history")
        
        if len(history_list) > 0:
            # Convert to DataFrame
            history = pd.DataFrame(history_list)
            print(f"Available metrics: {list(history.columns)}")
        else:
            # Fallback to regular history method
            history = run.history()
            print(f"Fetched {len(history)} steps via history()")
    except Exception as e:
        print(f"Error fetching history: {e}")
        history = pd.DataFrame()
    
    if len(history) == 0:
        print("No history data available!")
        return
    
    # Analyze gradient ascent/descent losses
    print("\nGradient Loss Analysis:")
    print("-" * 40)
    
    # Check for different possible column names
    ascent_col = None
    descent_col = None
    
    for col in ['gradient_ascent_loss', 'train/gradient_ascent_loss']:
        if col in history.columns:
            ascent_col = col
            break
    
    for col in ['gradient_descent_loss', 'train/gradient_descent_loss']:
        if col in history.columns:
            descent_col = col
            break
    
    if ascent_col and descent_col:
        # Get first and last non-null values
        ascent_losses = history[ascent_col].dropna()
        descent_losses = history[descent_col].dropna()
        
        if len(ascent_losses) > 0:
            first_ascent = ascent_losses.iloc[0]
            last_ascent = ascent_losses.iloc[-1]
            ascent_change = last_ascent - first_ascent
            ascent_pct_change = (ascent_change / first_ascent) * 100
            
            print(f"\nGradient ASCENT Loss:")
            print(f"  First value: {first_ascent:.6f}")
            print(f"  Last value:  {last_ascent:.6f}")
            print(f"  Change:      {ascent_change:.6f} ({ascent_pct_change:+.2f}%)")
            print(f"  Trend:       {'INCREASING ✅' if ascent_change > 0 else 'DECREASING ❌'}")
            
            if ascent_change <= 0:
                print("  ⚠️  WARNING: Ascent loss should be INCREASING for proper unlearning!")
        
        if len(descent_losses) > 0:
            first_descent = descent_losses.iloc[0]
            last_descent = descent_losses.iloc[-1]
            descent_change = last_descent - first_descent
            descent_pct_change = (descent_change / first_descent) * 100
            
            print(f"\nGradient DESCENT Loss:")
            print(f"  First value: {first_descent:.6f}")
            print(f"  Last value:  {last_descent:.6f}")
            print(f"  Change:      {descent_change:.6f} ({descent_pct_change:+.2f}%)")
            print(f"  Trend:       {'DECREASING ✅' if descent_change < 0 else 'INCREASING ❌'}")
        
        # Check divergence
        if len(ascent_losses) > 0 and len(descent_losses) > 0:
            print(f"\nDivergence Analysis:")
            print(f"  Losses moving in opposite directions: {'YES ✅' if ascent_change > 0 and descent_change < 0 else 'NO ❌'}")
    else:
        print("Gradient ascent/descent losses not found in metrics!")
        print(f"Available metrics: {list(history.columns)}")
    
    # Check sample distribution
    ascent_samples_col = 'train/gradient_ascent_samples' if 'train/gradient_ascent_samples' in history.columns else 'gradient_ascent_samples'
    descent_samples_col = 'train/gradient_descent_samples' if 'train/gradient_descent_samples' in history.columns else 'gradient_descent_samples'
    
    if ascent_samples_col in history.columns and descent_samples_col in history.columns:
        ascent_samples = history[ascent_samples_col].dropna()
        descent_samples = history[descent_samples_col].dropna()
        
        if len(ascent_samples) > 0 and len(descent_samples) > 0:
            avg_ascent = ascent_samples.mean()
            avg_descent = descent_samples.mean()
            total_samples = avg_ascent + avg_descent
            ascent_ratio = (avg_ascent / total_samples) * 100
            
            print(f"\nSample Distribution:")
            print(f"  Average ascent samples:  {avg_ascent:.0f} ({ascent_ratio:.2f}%)")
            print(f"  Average descent samples: {avg_descent:.0f} ({100-ascent_ratio:.2f}%)")
    
    # Check main loss trend
    loss_col = 'train/lm_loss' if 'train/lm_loss' in history.columns else 'loss'
    if loss_col in history.columns:
        losses = history[loss_col].dropna()
        if len(losses) > 0:
            first_loss = losses.iloc[0]
            last_loss = losses.iloc[-1]
            loss_change = last_loss - first_loss
            
            print(f"\nMain Loss:")
            print(f"  First value: {first_loss:.6f}")
            print(f"  Last value:  {last_loss:.6f}")
            print(f"  Change:      {loss_change:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    
    if ascent_col and len(ascent_losses) > 0:
        if ascent_change > 0:
            print("✅ Gradient ascent appears to be working correctly (loss increasing)")
        else:
            print("❌ PROBLEM: Gradient ascent loss is decreasing!")
            print("   Possible issues:")
            print("   - The fix might not be deployed in this run")
            print("   - The gradient_ascent_loss_scale might be too low")
            print("   - There might be another bug")
    else:
        print("⚠️  Cannot determine gradient ascent behavior - metrics not found")
    
    print("=" * 80)


if __name__ == "__main__":
    # Analyze the specified run
    run_path = "eleutherai/AISI/hqakruly"
    analyze_gradient_ascent_run(run_path)