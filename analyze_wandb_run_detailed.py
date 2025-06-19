#!/usr/bin/env python
"""
Detailed analysis of W&B run to check gradient ascent behavior over time.
"""

import wandb
import pandas as pd
import numpy as np

def analyze_gradient_ascent_run_detailed(run_path):
    """Analyze a W&B run with detailed gradient ascent behavior tracking"""
    
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
    
    # Get metrics history
    print("\nFetching detailed metrics history...")
    
    try:
        history = run.scan_history()
        history_list = list(history)
        print(f"Fetched {len(history_list)} steps of data")
        
        if len(history_list) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(history_list)
            print(f"Steps range: {df['_step'].min()} to {df['_step'].max()}")
        else:
            print("No history data!")
            return
    except Exception as e:
        print(f"Error fetching history: {e}")
        return
    
    # Analyze gradient ascent/descent losses in detail
    print("\nDetailed Gradient Loss Analysis:")
    print("-" * 40)
    
    # Filter to only rows with both ascent and descent losses
    ascent_col = 'train/gradient_ascent_loss'
    descent_col = 'train/gradient_descent_loss'
    
    if ascent_col in df.columns and descent_col in df.columns:
        # Create clean dataframe with both losses
        clean_df = df[['_step', ascent_col, descent_col]].dropna()
        
        if len(clean_df) > 0:
            # Calculate statistics in windows
            n_windows = 10
            window_size = len(clean_df) // n_windows
            
            print(f"\nAnalyzing {len(clean_df)} steps with both ascent and descent losses")
            print(f"Breaking into {n_windows} windows of ~{window_size} steps each\n")
            
            print("Window Analysis:")
            print(f"{'Window':<10} {'Steps':<15} {'Ascent Loss':<25} {'Descent Loss':<25} {'Ascent Trend':<15}")
            print("-" * 100)
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else len(clean_df)
                
                window_df = clean_df.iloc[start_idx:end_idx]
                
                step_range = f"{window_df['_step'].iloc[0]}-{window_df['_step'].iloc[-1]}"
                
                ascent_mean = window_df[ascent_col].mean()
                ascent_start = window_df[ascent_col].iloc[0]
                ascent_end = window_df[ascent_col].iloc[-1]
                ascent_change = ascent_end - ascent_start
                
                descent_mean = window_df[descent_col].mean()
                descent_start = window_df[descent_col].iloc[0]
                descent_end = window_df[descent_col].iloc[-1]
                descent_change = descent_end - descent_start
                
                ascent_trend = "↑" if ascent_change > 0 else "↓"
                
                print(f"{i+1:<10} {step_range:<15} "
                      f"{ascent_start:.4f} → {ascent_end:.4f} "
                      f"{descent_start:.4f} → {descent_end:.4f} "
                      f"{ascent_trend} {ascent_change:+.4f}")
            
            # Overall statistics
            print("\nOverall Statistics:")
            print("-" * 40)
            
            # Check if ascent loss ever increases
            ascent_losses = clean_df[ascent_col].values
            increasing_steps = 0
            for i in range(1, len(ascent_losses)):
                if ascent_losses[i] > ascent_losses[i-1]:
                    increasing_steps += 1
            
            print(f"Steps where ascent loss increased: {increasing_steps}/{len(ascent_losses)-1} ({increasing_steps/(len(ascent_losses)-1)*100:.1f}%)")
            
            # Calculate correlation
            correlation = clean_df[ascent_col].corr(clean_df[descent_col])
            print(f"Correlation between ascent and descent losses: {correlation:.4f}")
            
            # Find max and min points
            max_ascent_idx = clean_df[ascent_col].idxmax()
            min_ascent_idx = clean_df[ascent_col].idxmin()
            
            print(f"\nGradient Ascent Loss:")
            print(f"  Maximum: {clean_df.loc[max_ascent_idx, ascent_col]:.6f} at step {clean_df.loc[max_ascent_idx, '_step']}")
            print(f"  Minimum: {clean_df.loc[min_ascent_idx, ascent_col]:.6f} at step {clean_df.loc[min_ascent_idx, '_step']}")
            print(f"  First value: {clean_df[ascent_col].iloc[0]:.6f}")
            print(f"  Last value:  {clean_df[ascent_col].iloc[-1]:.6f}")
            print(f"  Overall change: {clean_df[ascent_col].iloc[-1] - clean_df[ascent_col].iloc[0]:.6f}")
            
            # Sample distribution analysis
            if 'train/gradient_ascent_samples' in df.columns:
                ascent_samples = df['train/gradient_ascent_samples'].dropna()
                descent_samples = df['train/gradient_descent_samples'].dropna()
                
                if len(ascent_samples) > 0:
                    print(f"\nSample Distribution Details:")
                    print(f"  Total ascent samples (sum): {ascent_samples.sum():.0f}")
                    print(f"  Total descent samples (sum): {descent_samples.sum():.0f}")
                    print(f"  Percentage of ascent samples: {ascent_samples.sum() / (ascent_samples.sum() + descent_samples.sum()) * 100:.2f}%")
    
    else:
        print("Gradient ascent/descent losses not found in metrics!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Analyze the specified run
    run_path = "eleutherai/AISI/azez8ylb"
    analyze_gradient_ascent_run_detailed(run_path)