#!/usr/bin/env python3
"""
Detailed analysis of gradient ascent loss curves from W&B runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from analyze_wandb_run import WandbRunAnalyzer
import argparse


def analyze_ga_loss_patterns(run_path):
    """Perform detailed analysis of GA loss patterns."""
    
    analyzer = WandbRunAnalyzer(run_path)
    df = analyzer.get_history()
    
    # Extract GA-related columns
    ga_actual_loss = df[['_step', 'train/ga_actual_loss']].dropna()
    ga_objective = df[['_step', 'train/ga_objective']].dropna()
    regular_loss = df[['_step', 'train/lm_loss']].dropna()
    
    print(f"\n{'='*80}")
    print(f"Gradient Ascent Loss Analysis for {analyzer.run.name}")
    print(f"{'='*80}")
    
    # Basic statistics
    print("\n## GA Loss Statistics")
    print(f"Total GA steps: {len(ga_actual_loss)}")
    print(f"GA step percentage: {len(ga_actual_loss) / len(df) * 100:.1f}%")
    
    if len(ga_actual_loss) > 0:
        print(f"\nGA Actual Loss:")
        print(f"  - Mean: {ga_actual_loss['train/ga_actual_loss'].mean():.6f}")
        print(f"  - Std: {ga_actual_loss['train/ga_actual_loss'].std():.6f}")
        print(f"  - Min: {ga_actual_loss['train/ga_actual_loss'].min():.6f}")
        print(f"  - Max: {ga_actual_loss['train/ga_actual_loss'].max():.6f}")
        print(f"  - Median: {ga_actual_loss['train/ga_actual_loss'].median():.6f}")
        
        # Trend analysis
        print("\n## Trend Analysis")
        
        # Linear regression on GA actual loss
        x = ga_actual_loss['_step'].values
        y = ga_actual_loss['train/ga_actual_loss'].values
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        print(f"GA Actual Loss Linear Trend:")
        print(f"  - Slope: {slope:.2e} (loss change per step)")
        print(f"  - R²: {r_value**2:.4f}")
        print(f"  - P-value: {p_value:.2e}")
        
        # Analyze GA intervals
        ga_steps = ga_actual_loss['_step'].values
        ga_intervals = np.diff(ga_steps)
        
        print(f"\n## GA Interval Analysis")
        print(f"Step intervals between GA iterations:")
        print(f"  - Mean interval: {ga_intervals.mean():.1f} steps")
        print(f"  - Std interval: {ga_intervals.std():.1f} steps")
        print(f"  - Min interval: {ga_intervals.min()} steps")
        print(f"  - Max interval: {ga_intervals.max()} steps")
        
        # Detect GA pattern (interval vs interleaved)
        unique_intervals = np.unique(ga_intervals)
        if len(unique_intervals) < 5 and ga_intervals.std() < 2:
            print(f"  - Pattern: Likely interleaved mode (consistent intervals)")
        else:
            print(f"  - Pattern: Likely interval mode (burst pattern)")
        
        # Compare GA loss to regular training loss
        print("\n## GA vs Regular Training Loss Comparison")
        
        # Find average training loss around GA steps
        comparison_data = []
        window = 50  # Look at ±50 steps around each GA step
        
        for ga_step in ga_steps[:100]:  # Analyze first 100 GA steps
            # Get regular losses in window around GA step
            mask = (regular_loss['_step'] >= ga_step - window) & \
                   (regular_loss['_step'] <= ga_step + window) & \
                   (regular_loss['_step'] != ga_step)  # Exclude the GA step itself
            
            window_losses = regular_loss[mask]['train/lm_loss'].values
            if len(window_losses) > 0:
                ga_loss_val = ga_actual_loss[ga_actual_loss['_step'] == ga_step]['train/ga_actual_loss'].values
                if len(ga_loss_val) > 0:
                    comparison_data.append({
                        'step': ga_step,
                        'ga_loss': ga_loss_val[0],
                        'avg_train_loss': window_losses.mean(),
                        'ratio': ga_loss_val[0] / window_losses.mean()
                    })
        
        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            print(f"  - Average GA/Training loss ratio: {comp_df['ratio'].mean():.3f}")
            print(f"  - Std of ratio: {comp_df['ratio'].std():.3f}")
        
        # Analyze GA loss evolution over time
        print("\n## GA Loss Evolution")
        
        # Split into quarters
        n_quarters = 4
        quarter_size = len(ga_actual_loss) // n_quarters
        
        for i in range(n_quarters):
            start_idx = i * quarter_size
            end_idx = (i + 1) * quarter_size if i < n_quarters - 1 else len(ga_actual_loss)
            
            quarter_data = ga_actual_loss.iloc[start_idx:end_idx]
            quarter_mean = quarter_data['train/ga_actual_loss'].mean()
            quarter_std = quarter_data['train/ga_actual_loss'].std()
            
            print(f"  Quarter {i+1} (steps {quarter_data['_step'].min()}-{quarter_data['_step'].max()}):")
            print(f"    - Mean: {quarter_mean:.6f}")
            print(f"    - Std: {quarter_std:.6f}")
    
    return df, ga_actual_loss, ga_objective, regular_loss


def plot_detailed_ga_analysis(df, ga_actual_loss, ga_objective, regular_loss, save_path=None):
    """Create detailed plots for GA loss analysis."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Detailed Gradient Ascent Loss Analysis', fontsize=16)
    
    # Plot 1: GA Actual Loss with trend
    ax = axes[0, 0]
    ax.scatter(ga_actual_loss['_step'], ga_actual_loss['train/ga_actual_loss'], 
               alpha=0.5, s=10, label='GA Actual Loss')
    
    # Add trend line
    z = np.polyfit(ga_actual_loss['_step'], ga_actual_loss['train/ga_actual_loss'], 1)
    p = np.poly1d(z)
    ax.plot(ga_actual_loss['_step'], p(ga_actual_loss['_step']), 
            "r--", alpha=0.8, label=f'Trend (slope={z[0]:.2e})')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('GA Actual Loss')
    ax.set_title('GA Actual Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: GA Objective (negated loss)
    ax = axes[0, 1]
    ax.scatter(ga_objective['_step'], ga_objective['train/ga_objective'], 
               alpha=0.5, s=10, color='darkred', label='GA Objective')
    ax.set_xlabel('Step')
    ax.set_ylabel('GA Objective (Negated Loss)')
    ax.set_title('GA Objective Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: GA Loss Distribution
    ax = axes[1, 0]
    ax.hist(ga_actual_loss['train/ga_actual_loss'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(ga_actual_loss['train/ga_actual_loss'].mean(), color='red', 
               linestyle='--', label=f'Mean: {ga_actual_loss["train/ga_actual_loss"].mean():.3f}')
    ax.set_xlabel('GA Actual Loss')
    ax.set_ylabel('Frequency')
    ax.set_title('GA Loss Distribution')
    ax.legend()
    
    # Plot 4: GA vs Regular Loss Comparison
    ax = axes[1, 1]
    
    # Sample points for clarity
    sample_size = min(1000, len(regular_loss))
    sample_indices = np.random.choice(len(regular_loss), sample_size, replace=False)
    
    ax.scatter(regular_loss.iloc[sample_indices]['_step'], 
               regular_loss.iloc[sample_indices]['train/lm_loss'], 
               alpha=0.3, s=5, label='Regular Loss', color='gray')
    ax.scatter(ga_actual_loss['_step'], ga_actual_loss['train/ga_actual_loss'], 
               alpha=0.7, s=20, label='GA Loss', color='red')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('GA Loss vs Regular Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: GA Interval Pattern
    ax = axes[2, 0]
    ga_steps = ga_actual_loss['_step'].values
    ga_intervals = np.diff(ga_steps)
    
    ax.plot(ga_intervals, alpha=0.7)
    ax.axhline(ga_intervals.mean(), color='red', linestyle='--', 
               label=f'Mean: {ga_intervals.mean():.1f}')
    ax.set_xlabel('GA Occurrence Index')
    ax.set_ylabel('Steps Between GA')
    ax.set_title('GA Interval Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Rolling statistics
    ax = axes[2, 1]
    window = 100
    if len(ga_actual_loss) > window:
        rolling_mean = ga_actual_loss.set_index('_step')['train/ga_actual_loss'].rolling(window=window).mean()
        rolling_std = ga_actual_loss.set_index('_step')['train/ga_actual_loss'].rolling(window=window).std()
        
        ax.plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean (window={window})', color='blue')
        ax.fill_between(rolling_std.index, 
                        rolling_mean.values - rolling_std.values,
                        rolling_mean.values + rolling_std.values,
                        alpha=0.3, color='blue', label='±1 STD')
        ax.set_xlabel('Step')
        ax.set_ylabel('GA Loss')
        ax.set_title('GA Loss Rolling Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDetailed analysis plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detailed GA loss analysis')
    parser.add_argument('run_path', type=str, help='W&B run path (entity/project/run_id)')
    parser.add_argument('--save-plot', type=str, help='Save analysis plot to file')
    
    args = parser.parse_args()
    
    # Perform analysis
    df, ga_actual_loss, ga_objective, regular_loss = analyze_ga_loss_patterns(args.run_path)
    
    # Create plots
    if len(ga_actual_loss) > 0:
        plot_detailed_ga_analysis(df, ga_actual_loss, ga_objective, regular_loss, args.save_plot)
    else:
        print("\nNo GA losses found in this run!")


if __name__ == "__main__":
    main()