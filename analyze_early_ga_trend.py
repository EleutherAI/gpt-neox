#!/usr/bin/env python3
"""
Analyze the early training GA loss trend (up to 570 steps) to understand initial behavior.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analyze_wandb_run import WandbRunAnalyzer
import argparse


def analyze_early_ga_trend(run_path, max_step=570):
    """Focus analysis on early GA training steps."""
    
    analyzer = WandbRunAnalyzer(run_path)
    df = analyzer.get_history()
    
    # Filter for early steps
    early_df = df[df['_step'] <= max_step].copy()
    
    # Extract GA metrics
    ga_actual_loss = early_df[['_step', 'train/ga_actual_loss']].dropna()
    regular_loss = early_df[['_step', 'train/lm_loss']].dropna()
    
    print(f"\n{'='*80}")
    print(f"Early Training GA Analysis (Steps 0-{max_step})")
    print(f"{'='*80}")
    
    if len(ga_actual_loss) == 0:
        print("No GA steps found in early training!")
        return None, None, None
    
    # Identify first GA occurrence
    first_ga_step = ga_actual_loss['_step'].min()
    print(f"\nFirst GA step occurs at: {first_ga_step}")
    
    # Early GA statistics
    early_ga_losses = ga_actual_loss['train/ga_actual_loss'].values
    print(f"\nEarly GA Loss Statistics:")
    print(f"  - Number of GA steps: {len(early_ga_losses)}")
    print(f"  - First GA loss: {early_ga_losses[0]:.6f}")
    print(f"  - Last GA loss (before step {max_step}): {early_ga_losses[-1]:.6f}")
    print(f"  - Change: {early_ga_losses[-1] - early_ga_losses[0]:.6f}")
    print(f"  - Percent change: {((early_ga_losses[-1] - early_ga_losses[0]) / early_ga_losses[0] * 100):.1f}%")
    
    # Analyze trend segments
    if len(early_ga_losses) > 10:
        # First 5 GA steps
        print(f"\nFirst 5 GA steps:")
        for i in range(min(5, len(ga_actual_loss))):
            step = ga_actual_loss.iloc[i]['_step']
            loss = ga_actual_loss.iloc[i]['train/ga_actual_loss']
            print(f"  Step {step}: {loss:.6f}")
        
        # Check if loss is increasing or decreasing initially
        first_5_losses = early_ga_losses[:5]
        if len(first_5_losses) >= 2:
            initial_trend = "INCREASING" if first_5_losses[-1] > first_5_losses[0] else "DECREASING"
            print(f"\nInitial trend (first 5 GA steps): {initial_trend}")
        
        # Look for trend reversal
        print(f"\nTrend Analysis:")
        
        # Split into segments
        segment_size = len(early_ga_losses) // 3
        if segment_size > 0:
            seg1 = early_ga_losses[:segment_size]
            seg2 = early_ga_losses[segment_size:2*segment_size]
            seg3 = early_ga_losses[2*segment_size:]
            
            print(f"  Segment 1 (early): mean={np.mean(seg1):.6f}, std={np.std(seg1):.6f}")
            print(f"  Segment 2 (middle): mean={np.mean(seg2):.6f}, std={np.std(seg2):.6f}")
            print(f"  Segment 3 (late): mean={np.mean(seg3):.6f}, std={np.std(seg3):.6f}")
    
    # Find peaks and valleys
    if len(early_ga_losses) > 3:
        print(f"\nPeak Analysis:")
        max_idx = np.argmax(early_ga_losses)
        max_step = ga_actual_loss.iloc[max_idx]['_step']
        print(f"  - Maximum GA loss: {early_ga_losses[max_idx]:.6f} at step {max_step}")
        
        min_idx = np.argmin(early_ga_losses)
        min_step = ga_actual_loss.iloc[min_idx]['_step']
        print(f"  - Minimum GA loss: {early_ga_losses[min_idx]:.6f} at step {min_step}")
        
        if max_step < min_step:
            print(f"  - Pattern: Loss peaked early then decreased (peak before valley)")
        else:
            print(f"  - Pattern: Loss decreased then increased (valley before peak)")
    
    # Compare with regular training loss at same steps
    print(f"\nComparison with Regular Training Loss:")
    
    # Get regular loss values at GA steps
    ga_steps = ga_actual_loss['_step'].values
    comparison_data = []
    
    for ga_step in ga_steps[:10]:  # First 10 GA steps
        # Find nearest regular loss
        regular_at_step = regular_loss[regular_loss['_step'] == ga_step]
        if len(regular_at_step) > 0:
            continue  # Skip if this is a GA step
        
        # Find closest regular training step
        nearest_regular = regular_loss.iloc[(regular_loss['_step'] - ga_step).abs().argsort()[:1]]
        if len(nearest_regular) > 0:
            reg_step = nearest_regular.iloc[0]['_step']
            reg_loss = nearest_regular.iloc[0]['train/lm_loss']
            ga_loss = ga_actual_loss[ga_actual_loss['_step'] == ga_step].iloc[0]['train/ga_actual_loss']
            
            comparison_data.append({
                'ga_step': ga_step,
                'ga_loss': ga_loss,
                'nearest_regular_step': reg_step,
                'regular_loss': reg_loss,
                'difference': ga_loss - reg_loss
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print(f"  - Average GA loss: {comp_df['ga_loss'].mean():.6f}")
        print(f"  - Average regular loss: {comp_df['regular_loss'].mean():.6f}")
        print(f"  - Average difference (GA - regular): {comp_df['difference'].mean():.6f}")
    
    return early_df, ga_actual_loss, regular_loss


def plot_early_ga_trend(early_df, ga_actual_loss, regular_loss, max_step=570, save_path=None):
    """Create detailed plots for early GA trend analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Early Training GA Analysis (Steps 0-{max_step})', fontsize=16)
    
    # Plot 1: GA Actual Loss with annotations
    ax = axes[0, 0]
    ga_steps = ga_actual_loss['_step'].values
    ga_losses = ga_actual_loss['train/ga_actual_loss'].values
    
    ax.plot(ga_steps, ga_losses, 'ro-', markersize=6, label='GA Actual Loss')
    
    # Annotate first and last points
    if len(ga_losses) > 0:
        ax.annotate(f'Start: {ga_losses[0]:.3f}', 
                   xy=(ga_steps[0], ga_losses[0]), 
                   xytext=(ga_steps[0]+20, ga_losses[0]+0.1),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.annotate(f'End: {ga_losses[-1]:.3f}', 
                   xy=(ga_steps[-1], ga_losses[-1]), 
                   xytext=(ga_steps[-1]-50, ga_losses[-1]+0.1),
                   arrowprops=dict(arrowstyle='->', color='red'))
    
    # Mark peak if exists
    if len(ga_losses) > 3:
        max_idx = np.argmax(ga_losses)
        ax.plot(ga_steps[max_idx], ga_losses[max_idx], 'g*', markersize=15, label=f'Peak: {ga_losses[max_idx]:.3f}')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('GA Actual Loss')
    ax.set_title('GA Loss Evolution in Early Training')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Combined view with regular loss
    ax = axes[0, 1]
    
    # Plot regular loss as background
    reg_steps = regular_loss['_step'].values
    reg_losses = regular_loss['train/lm_loss'].values
    ax.plot(reg_steps, reg_losses, 'b-', alpha=0.3, linewidth=2, label='Regular Training Loss')
    
    # Overlay GA loss
    ax.plot(ga_steps, ga_losses, 'ro-', markersize=6, label='GA Actual Loss')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('GA vs Regular Training Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: GA loss change rate
    ax = axes[1, 0]
    if len(ga_losses) > 1:
        # Calculate change between consecutive GA steps
        ga_changes = np.diff(ga_losses)
        ga_change_steps = ga_steps[1:]
        
        colors = ['red' if x < 0 else 'green' for x in ga_changes]
        ax.bar(ga_change_steps, ga_changes, color=colors, alpha=0.7, width=5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add cumulative change line
        cumulative_change = np.cumsum(ga_changes)
        ax2 = ax.twinx()
        ax2.plot(ga_change_steps, cumulative_change, 'k--', linewidth=2, label='Cumulative Change')
        ax2.set_ylabel('Cumulative Change', color='k')
        ax2.tick_params(axis='y', labelcolor='k')
        
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss Change (Step-to-Step)')
    ax.set_title('GA Loss Change Rate')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Moving average
    ax = axes[1, 1]
    if len(ga_losses) > 5:
        # Simple moving average
        window = min(5, len(ga_losses) // 3)
        ma = pd.Series(ga_losses).rolling(window=window, center=True).mean()
        
        ax.plot(ga_steps, ga_losses, 'ro', alpha=0.5, label='GA Loss')
        ax.plot(ga_steps, ma, 'r-', linewidth=3, label=f'{window}-step Moving Avg')
        
        # Add trend line
        z = np.polyfit(ga_steps, ga_losses, 1)
        p = np.poly1d(z)
        ax.plot(ga_steps, p(ga_steps), 'g--', linewidth=2, 
                label=f'Trend: {"↑" if z[0] > 0 else "↓"} {abs(z[0]):.2e}/step')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('GA Loss')
    ax.set_title('GA Loss Trend Analysis')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze early GA training trends')
    parser.add_argument('run_path', type=str, help='W&B run path')
    parser.add_argument('--max-step', type=int, default=570, help='Maximum step to analyze')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    
    args = parser.parse_args()
    
    # Perform analysis
    early_df, ga_actual_loss, regular_loss = analyze_early_ga_trend(args.run_path, args.max_step)
    
    # Create plots if data exists
    if ga_actual_loss is not None and len(ga_actual_loss) > 0:
        plot_early_ga_trend(early_df, ga_actual_loss, regular_loss, args.max_step, args.save_plot)


if __name__ == "__main__":
    main()