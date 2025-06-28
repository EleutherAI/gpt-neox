#!/usr/bin/env python3
"""
Analyze Weights & Biases training runs for GPT-NeoX experiments.
This script fetches metrics, analyzes training performance, and helps diagnose crashes.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import json
from typing import Dict, List, Optional, Tuple
import wandb
from wandb.apis.public import Api
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class WandbRunAnalyzer:
    """Analyze W&B runs for training metrics, gradient ascent metrics, and crashes."""
    
    def __init__(self, run_path: str, api_key: Optional[str] = None):
        """
        Initialize the analyzer with a run path.
        
        Args:
            run_path: Full path to the run (entity/project/run_id)
            api_key: Optional W&B API key (will use environment variable if not provided)
        """
        if api_key:
            wandb.login(key=api_key)
        
        self.api = Api()
        self.run_path = run_path
        
        # Parse the run path
        parts = run_path.split('/')
        if len(parts) != 3:
            raise ValueError(f"Invalid run path format. Expected 'entity/project/run_id', got '{run_path}'")
        
        self.entity, self.project, self.run_id = parts
        
        # Fetch the run
        try:
            self.run = self.api.run(run_path)
        except Exception as e:
            print(f"Error fetching run: {e}")
            raise
        
        # Cache for metrics data
        self._history_df = None
        self._system_metrics_df = None
    
    def get_run_info(self) -> Dict:
        """Get basic information about the run."""
        info = {
            'name': self.run.name,
            'id': self.run.id,
            'state': self.run.state,
            'created_at': self.run.created_at,
            'runtime': self.run.summary.get('_runtime', 0),
            'step': self.run.summary.get('_step', 0),
            'tags': self.run.tags,
            'url': self.run.url
        }
        
        # Add config info
        info['config'] = dict(self.run.config)
        
        return info
    
    def get_history(self, samples: int = None) -> pd.DataFrame:
        """
        Get the full history of metrics for the run.
        
        Args:
            samples: Number of samples to retrieve (None for all)
        """
        if self._history_df is None or samples is not None:
            history = self.run.history(samples=samples or 10000)
            self._history_df = pd.DataFrame(history)
        return self._history_df
    
    def get_system_metrics(self) -> pd.DataFrame:
        """Get system metrics like GPU usage, memory, etc."""
        if self._system_metrics_df is None:
            system_history = self.run.history(stream='system')
            self._system_metrics_df = pd.DataFrame(system_history)
        return self._system_metrics_df
    
    def analyze_training_metrics(self) -> Dict:
        """Analyze training loss and related metrics."""
        df = self.get_history()
        
        metrics = {}
        
        # Training loss analysis
        if 'train/lm_loss' in df.columns:
            loss_series = df['train/lm_loss'].dropna()
            metrics['training_loss'] = {
                'final': loss_series.iloc[-1] if len(loss_series) > 0 else None,
                'min': loss_series.min(),
                'max': loss_series.max(),
                'mean': loss_series.mean(),
                'std': loss_series.std(),
                'num_steps': len(loss_series)
            }
        
        # Learning rate
        if 'train/learning_rate' in df.columns:
            lr_series = df['train/learning_rate'].dropna()
            metrics['learning_rate'] = {
                'initial': lr_series.iloc[0] if len(lr_series) > 0 else None,
                'final': lr_series.iloc[-1] if len(lr_series) > 0 else None,
                'min': lr_series.min(),
                'max': lr_series.max()
            }
        
        # Training speed metrics
        speed_metrics = {}
        for metric_name in ['runtime/samples_per_sec', 'runtime/tokens_per_sec', 
                           'runtime/iteration_time', 'runtime/flops_per_sec_per_gpu']:
            if metric_name in df.columns:
                series = df[metric_name].dropna()
                if len(series) > 0:
                    speed_metrics[metric_name] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max()
                    }
        metrics['speed_metrics'] = speed_metrics
        
        return metrics
    
    def analyze_gradient_ascent_metrics(self) -> Dict:
        """Analyze gradient ascent specific metrics."""
        df = self.get_history()
        
        ga_metrics = {}
        
        # GA loss metrics
        ga_loss_columns = ['train/ga_actual_loss', 'train/ga_objective', 'train/ga_lr_scale']
        for col in ga_loss_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    ga_metrics[col] = {
                        'count': len(series),
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max()
                    }
        
        # GA iteration counts
        ga_iter_columns = ['train/ga_iterations_this_step', 'train/total_ga_iterations']
        for col in ga_iter_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    ga_metrics[col] = {
                        'final': series.iloc[-1],
                        'max': series.max()
                    }
        
        # Detect GA mode from config
        config = dict(self.run.config)
        ga_mode = config.get('ga_mode', 'none')
        ga_interval = config.get('ga_interval', None)
        ga_iters = config.get('ga_iters', None)
        ga_interleave_ratio = config.get('ga_interleave_ratio', None)
        
        ga_metrics['config'] = {
            'mode': ga_mode,
            'interval': ga_interval,
            'iters': ga_iters,
            'interleave_ratio': ga_interleave_ratio
        }
        
        return ga_metrics
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """
        Get the last N lines of logs from the run.
        
        Args:
            lines: Number of log lines to retrieve
        """
        try:
            # Get log file
            files = self.run.files()
            log_files = [f for f in files if f.name.endswith('.log') or f.name == 'output.log']
            
            if not log_files:
                return ["No log files found for this run"]
            
            # Get the main log file
            log_file = log_files[0]
            log_content = log_file.download(replace=True).read()
            
            # Handle both string and bytes
            if isinstance(log_content, bytes):
                log_content = log_content.decode('utf-8', errors='ignore')
            
            # Split into lines and get last N
            log_lines = log_content.split('\n')
            return log_lines[-lines:]
            
        except Exception as e:
            return [f"Error fetching logs: {str(e)}"]
    
    def diagnose_crash(self) -> Dict:
        """Diagnose potential crashes or issues in the run."""
        diagnosis = {
            'state': self.run.state,
            'exit_code': self.run.summary.get('exitcode', None),
            'runtime': self.run.summary.get('_runtime', 0),
            'last_step': self.run.summary.get('_step', 0)
        }
        
        # Check for NaN losses
        df = self.get_history()
        if 'train/lm_loss' in df.columns:
            loss_series = df['train/lm_loss']
            nan_count = loss_series.isna().sum()
            inf_count = np.isinf(loss_series.fillna(0)).sum()
            
            diagnosis['loss_issues'] = {
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'has_issues': nan_count > 0 or inf_count > 0
            }
        
        # Check system metrics for OOM
        try:
            system_df = self.get_system_metrics()
            if not system_df.empty:
                # Look for GPU memory percentage columns
                gpu_mem_cols = [col for col in system_df.columns 
                               if 'gpu' in col.lower() and 'memory' in col.lower() and 'percent' in col.lower()]
                if not gpu_mem_cols:
                    # Try alternative naming
                    gpu_mem_cols = [col for col in system_df.columns 
                                   if 'gpu' in col.lower() and 'memory' in col.lower()]
                
                if gpu_mem_cols:
                    # Get max GPU memory usage across all GPUs and time
                    max_values = []
                    for col in gpu_mem_cols:
                        col_max = system_df[col].dropna().max()
                        if not np.isnan(col_max) and col_max < 100:  # Sanity check
                            max_values.append(col_max)
                    
                    if max_values:
                        diagnosis['max_gpu_memory_percent'] = max(max_values)
        except:
            pass
        
        # Get last logs for error messages
        last_logs = self.get_logs(lines=50)
        error_keywords = ['error', 'exception', 'traceback', 'cuda out of memory', 
                         'nan', 'inf', 'overflow', 'assert']
        
        errors_found = []
        for line in last_logs:
            line_lower = line.lower()
            for keyword in error_keywords:
                if keyword in line_lower:
                    errors_found.append(line.strip())
                    break
        
        diagnosis['error_lines'] = errors_found[-10:]  # Last 10 error lines
        
        return diagnosis
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Plot training curves including loss and gradient ascent metrics."""
        df = self.get_history()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves for {self.run.name}', fontsize=16)
        
        # Plot 1: Training Loss
        if 'train/lm_loss' in df.columns:
            ax = axes[0, 0]
            loss_data = df[['_step', 'train/lm_loss']].dropna()
            ax.plot(loss_data['_step'], loss_data['train/lm_loss'], label='LM Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 2: Learning Rate
        if 'train/learning_rate' in df.columns:
            ax = axes[0, 1]
            lr_data = df[['_step', 'train/learning_rate']].dropna()
            ax.plot(lr_data['_step'], lr_data['train/learning_rate'], label='Learning Rate', color='orange')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Plot 3: GA Metrics (if available)
        ax = axes[1, 0]
        ga_plotted = False
        if 'train/ga_actual_loss' in df.columns:
            ga_data = df[['_step', 'train/ga_actual_loss']].dropna()
            if not ga_data.empty:
                ax.plot(ga_data['_step'], ga_data['train/ga_actual_loss'], 
                       label='GA Actual Loss', color='red', alpha=0.7)
                ga_plotted = True
        
        if 'train/ga_objective' in df.columns:
            ga_obj_data = df[['_step', 'train/ga_objective']].dropna()
            if not ga_obj_data.empty:
                ax.plot(ga_obj_data['_step'], ga_obj_data['train/ga_objective'], 
                       label='GA Objective', color='darkred', alpha=0.7)
                ga_plotted = True
        
        if ga_plotted:
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Gradient Ascent Metrics')
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No GA metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Plot 4: Training Speed
        ax = axes[1, 1]
        if 'runtime/samples_per_sec' in df.columns:
            speed_data = df[['_step', 'runtime/samples_per_sec']].dropna()
            if not speed_data.empty:
                ax.plot(speed_data['_step'], speed_data['runtime/samples_per_sec'], 
                       label='Samples/sec', color='green')
                ax.set_xlabel('Step')
                ax.set_ylabel('Samples per Second')
                ax.set_title('Training Speed')
                ax.grid(True, alpha=0.3)
                ax.legend()
        else:
            ax.text(0.5, 0.5, 'No speed metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report of the run analysis."""
        report = []
        report.append("="*80)
        report.append(f"W&B Run Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Basic info
        info = self.get_run_info()
        report.append("\n## Run Information")
        report.append(f"- Name: {info['name']}")
        report.append(f"- ID: {info['id']}")
        report.append(f"- State: {info['state']}")
        report.append(f"- Runtime: {info['runtime']/3600:.2f} hours")
        report.append(f"- Total Steps: {info['step']}")
        report.append(f"- URL: {info['url']}")
        
        # Training metrics
        report.append("\n## Training Metrics")
        train_metrics = self.analyze_training_metrics()
        
        if 'training_loss' in train_metrics:
            loss = train_metrics['training_loss']
            report.append(f"\n### Training Loss")
            report.append(f"- Final: {loss['final']:.6f}" if loss['final'] else "- Final: N/A")
            report.append(f"- Min: {loss['min']:.6f}")
            report.append(f"- Mean ± Std: {loss['mean']:.6f} ± {loss['std']:.6f}")
        
        if 'learning_rate' in train_metrics:
            lr = train_metrics['learning_rate']
            report.append(f"\n### Learning Rate")
            report.append(f"- Initial: {lr['initial']:.2e}" if lr['initial'] else "- Initial: N/A")
            report.append(f"- Final: {lr['final']:.2e}" if lr['final'] else "- Final: N/A")
        
        if 'speed_metrics' in train_metrics and train_metrics['speed_metrics']:
            report.append(f"\n### Training Speed")
            for metric, values in train_metrics['speed_metrics'].items():
                metric_name = metric.split('/')[-1].replace('_', ' ').title()
                report.append(f"- {metric_name}: {values['mean']:.2f} ± {values['std']:.2f}")
        
        # Gradient Ascent metrics
        ga_metrics = self.analyze_gradient_ascent_metrics()
        if ga_metrics and ga_metrics['config']['mode'] != 'none':
            report.append("\n## Gradient Ascent Analysis")
            report.append(f"- Mode: {ga_metrics['config']['mode']}")
            report.append(f"- Interval: {ga_metrics['config']['interval']}")
            report.append(f"- Iterations per GA: {ga_metrics['config']['iters']}")
            
            for metric_name, values in ga_metrics.items():
                if isinstance(values, dict) and 'mean' in values:
                    clean_name = metric_name.replace('train/', '').replace('_', ' ').title()
                    report.append(f"\n### {clean_name}")
                    report.append(f"- Mean: {values['mean']:.6f}")
                    report.append(f"- Std: {values['std']:.6f}")
                    report.append(f"- Count: {values['count']}")
        
        # Crash diagnosis
        report.append("\n## Crash/Issue Diagnosis")
        diagnosis = self.diagnose_crash()
        report.append(f"- Exit Code: {diagnosis['exit_code']}")
        
        if 'loss_issues' in diagnosis:
            issues = diagnosis['loss_issues']
            report.append(f"- NaN Loss Count: {issues['nan_count']}")
            report.append(f"- Inf Loss Count: {issues['inf_count']}")
        
        if 'max_gpu_memory_percent' in diagnosis:
            report.append(f"- Max GPU Memory: {diagnosis['max_gpu_memory_percent']:.1f}%")
        
        if diagnosis['error_lines']:
            report.append("\n### Error Lines Found:")
            for error in diagnosis['error_lines'][-5:]:  # Last 5 errors
                report.append(f"  - {error}")
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze W&B runs for GPT-NeoX training')
    parser.add_argument('run_path', type=str, 
                       help='Full W&B run path (entity/project/run_id)')
    parser.add_argument('--api-key', type=str, default=None,
                       help='W&B API key (uses environment variable if not provided)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and display training curves')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to specified path')
    parser.add_argument('--logs', type=int, default=0,
                       help='Number of log lines to print (0 to skip)')
    parser.add_argument('--report', type=str, default=None,
                       help='Save analysis report to specified file')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = WandbRunAnalyzer(args.run_path, args.api_key)
        
        # Generate and print report
        report = analyzer.generate_report()
        print(report)
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report}")
        
        # Show logs if requested
        if args.logs > 0:
            print(f"\n\n{'='*80}")
            print(f"Last {args.logs} log lines:")
            print('='*80)
            logs = analyzer.get_logs(args.logs)
            for line in logs:
                print(line)
        
        # Generate plots if requested
        if args.plot or args.save_plot:
            analyzer.plot_training_curves(args.save_plot)
        
    except Exception as e:
        print(f"Error analyzing run: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())