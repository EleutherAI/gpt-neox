#!/usr/bin/env python3
"""
Analyze W&B run eleutherai/AISI/9vbuhj4o with focus on:
1. Training metrics (loss trends)
2. Gradient ascent metrics (GA loss, iterations, learning rates)
3. Training time metrics
4. Any crashes or issues
5. Key configuration parameters
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


class RunAnalyzer:
    """Analyze W&B run 9vbuhj4o for training metrics, gradient ascent metrics, and crashes."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the analyzer for the specific run."""
        if api_key:
            wandb.login(key=api_key)
        
        self.api = Api()
        self.run_path = "eleutherai/AISI/9vbuhj4o"
        
        # Fetch the run
        try:
            self.run = self.api.run(self.run_path)
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
        """Get the full history of metrics for the run."""
        if self._history_df is None or samples is not None:
            history = self.run.history(samples=samples or 50000)
            self._history_df = pd.DataFrame(history)
        return self._history_df
    
    def get_system_metrics(self) -> pd.DataFrame:
        """Get system metrics like GPU usage, memory, etc."""
        if self._system_metrics_df is None:
            system_history = self.run.history(stream='system')
            self._system_metrics_df = pd.DataFrame(system_history)
        return self._system_metrics_df
    
    def analyze_training_metrics(self) -> Dict:
        """Analyze training loss and related metrics with focus on trends."""
        df = self.get_history()
        
        metrics = {}
        
        # Training loss analysis with trends
        if 'train/lm_loss' in df.columns:
            loss_series = df['train/lm_loss'].dropna()
            metrics['training_loss'] = {
                'final': loss_series.iloc[-1] if len(loss_series) > 0 else None,
                'initial': loss_series.iloc[0] if len(loss_series) > 0 else None,
                'min': loss_series.min(),
                'max': loss_series.max(),
                'mean': loss_series.mean(),
                'std': loss_series.std(),
                'num_steps': len(loss_series)
            }
            
            # Calculate loss trend (improvement)
            if len(loss_series) > 1:
                metrics['training_loss']['improvement'] = loss_series.iloc[0] - loss_series.iloc[-1]
                metrics['training_loss']['improvement_percent'] = ((loss_series.iloc[0] - loss_series.iloc[-1]) / loss_series.iloc[0]) * 100
                
                # Moving average to detect trends
                if len(loss_series) > 100:
                    ma_100 = loss_series.rolling(window=100).mean()
                    recent_trend = ma_100.iloc[-100:].mean() - ma_100.iloc[-200:-100].mean()
                    metrics['training_loss']['recent_trend'] = recent_trend
        
        # Learning rate analysis
        if 'train/learning_rate' in df.columns:
            lr_series = df['train/learning_rate'].dropna()
            metrics['learning_rate'] = {
                'initial': lr_series.iloc[0] if len(lr_series) > 0 else None,
                'final': lr_series.iloc[-1] if len(lr_series) > 0 else None,
                'min': lr_series.min(),
                'max': lr_series.max(),
                'mean': lr_series.mean()
            }
        
        # Gradient norms
        grad_norm_cols = ['train/grad_norm', 'train/grad_norm_before_clip']
        for col in grad_norm_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    metrics[col] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'max': series.max(),
                        'spikes': len(series[series > series.mean() + 3 * series.std()])
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
        """Analyze gradient ascent specific metrics in detail."""
        df = self.get_history()
        
        ga_metrics = {}
        
        # GA loss metrics with detailed statistics
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
                        'max': series.max(),
                        'median': series.median(),
                        'q25': series.quantile(0.25),
                        'q75': series.quantile(0.75)
                    }
                    
                    # Analyze trend
                    if len(series) > 10:
                        early_mean = series.iloc[:len(series)//3].mean()
                        late_mean = series.iloc[-len(series)//3:].mean()
                        ga_metrics[col]['trend'] = late_mean - early_mean
        
        # GA iteration analysis
        ga_iter_columns = ['train/ga_iterations_this_step', 'train/total_ga_iterations']
        for col in ga_iter_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    ga_metrics[col] = {
                        'final': series.iloc[-1],
                        'max': series.max(),
                        'total_ga_steps': len(series[series > 0])
                    }
        
        # GA learning rates
        ga_lr_columns = ['train/ga_lr', 'train/ga_lr_scaled']
        for col in ga_lr_columns:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) > 0:
                    ga_metrics[col] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'min': series.min(),
                        'max': series.max()
                    }
        
        # Detect GA mode and configuration
        config = dict(self.run.config)
        ga_config = {
            'mode': config.get('ga_mode', 'none'),
            'interval': config.get('ga_interval', None),
            'iters': config.get('ga_iters', None),
            'interleave_ratio': config.get('ga_interleave_ratio', None),
            'lr': config.get('ga_lr', None),
            'lr_scaling': config.get('ga_lr_scaling', None),
            'objective': config.get('ga_objective', None),
            'loss_scale': config.get('ga_loss_scale', None)
        }
        ga_metrics['config'] = ga_config
        
        # Calculate GA efficiency metrics
        if 'train/ga_actual_loss' in ga_metrics and 'train/lm_loss' in df.columns:
            ga_loss = ga_metrics['train/ga_actual_loss']['mean']
            normal_loss = df['train/lm_loss'].mean()
            ga_metrics['efficiency'] = {
                'loss_increase_ratio': ga_loss / normal_loss if normal_loss > 0 else None,
                'loss_difference': ga_loss - normal_loss
            }
        
        return ga_metrics
    
    def analyze_training_time(self) -> Dict:
        """Analyze training time metrics and efficiency."""
        df = self.get_history()
        info = self.get_run_info()
        
        time_metrics = {
            'total_runtime_hours': info['runtime'] / 3600,
            'total_steps': info['step']
        }
        
        # Calculate time per step
        if time_metrics['total_steps'] > 0:
            time_metrics['avg_seconds_per_step'] = info['runtime'] / time_metrics['total_steps']
            time_metrics['steps_per_hour'] = time_metrics['total_steps'] / (info['runtime'] / 3600) if info['runtime'] > 0 else 0
        
        # Iteration time analysis
        if 'runtime/iteration_time' in df.columns:
            iter_time = df['runtime/iteration_time'].dropna()
            if len(iter_time) > 0:
                time_metrics['iteration_time'] = {
                    'mean': iter_time.mean(),
                    'std': iter_time.std(),
                    'min': iter_time.min(),
                    'max': iter_time.max(),
                    'slowdowns': len(iter_time[iter_time > iter_time.mean() + 2 * iter_time.std()])
                }
        
        # GA time overhead
        if 'train/ga_iterations_this_step' in df.columns:
            ga_steps = df[df['train/ga_iterations_this_step'] > 0]
            non_ga_steps = df[df['train/ga_iterations_this_step'] == 0]
            
            if len(ga_steps) > 0 and len(non_ga_steps) > 0 and 'runtime/iteration_time' in df.columns:
                ga_time = ga_steps['runtime/iteration_time'].mean()
                non_ga_time = non_ga_steps['runtime/iteration_time'].mean()
                time_metrics['ga_overhead'] = {
                    'ga_step_time': ga_time,
                    'normal_step_time': non_ga_time,
                    'overhead_ratio': ga_time / non_ga_time if non_ga_time > 0 else None
                }
        
        return time_metrics
    
    def get_key_configuration(self) -> Dict:
        """Extract key configuration parameters."""
        config = dict(self.run.config)
        
        key_params = {
            # Model architecture
            'model': {
                'hidden_size': config.get('hidden_size'),
                'num_layers': config.get('num_layers'),
                'num_attention_heads': config.get('num_attention_heads'),
                'seq_length': config.get('seq_length'),
                'vocab_size': config.get('padded_vocab_size', config.get('vocab_size'))
            },
            # Training configuration
            'training': {
                'batch_size': config.get('train_batch_size'),
                'micro_batch_size': config.get('train_micro_batch_size_per_gpu'),
                'gradient_accumulation_steps': config.get('gradient_accumulation_steps'),
                'learning_rate': config.get('lr'),
                'min_lr': config.get('min_lr'),
                'warmup_steps': config.get('warmup', config.get('lr_decay_iters')),
                'total_steps': config.get('train_iters'),
                'optimizer': config.get('optimizer_type'),
                'precision': config.get('precision', config.get('fp16', {}).get('enabled') if config.get('fp16') else None)
            },
            # Gradient Ascent configuration
            'gradient_ascent': {
                'mode': config.get('ga_mode'),
                'interval': config.get('ga_interval'),
                'iterations': config.get('ga_iters'),
                'learning_rate': config.get('ga_lr'),
                'lr_scaling': config.get('ga_lr_scaling'),
                'objective': config.get('ga_objective'),
                'loss_scale': config.get('ga_loss_scale'),
                'interleave_ratio': config.get('ga_interleave_ratio')
            },
            # Parallelism
            'parallelism': {
                'data_parallel': config.get('data_parallel_size'),
                'tensor_parallel': config.get('model_parallel_size'),
                'pipeline_parallel': config.get('pipe_parallel_size'),
                'zero_stage': config.get('zero_optimization', {}).get('stage')
            }
        }
        
        return key_params
    
    def diagnose_crash(self) -> Dict:
        """Diagnose potential crashes or issues in the run."""
        diagnosis = {
            'state': self.run.state,
            'exit_code': self.run.summary.get('exitcode', None),
            'runtime': self.run.summary.get('_runtime', 0),
            'last_step': self.run.summary.get('_step', 0)
        }
        
        # Check for NaN/Inf losses
        df = self.get_history()
        if 'train/lm_loss' in df.columns:
            loss_series = df['train/lm_loss']
            nan_count = loss_series.isna().sum()
            inf_count = np.isinf(loss_series.fillna(0)).sum()
            
            diagnosis['loss_issues'] = {
                'nan_count': int(nan_count),
                'inf_count': int(inf_count),
                'has_issues': nan_count > 0 or inf_count > 0,
                'first_nan_step': df[loss_series.isna()]['_step'].min() if nan_count > 0 else None
            }
            
            # Check for loss spikes
            if len(loss_series.dropna()) > 10:
                loss_diff = loss_series.diff().abs()
                spikes = loss_diff[loss_diff > loss_series.std() * 3]
                diagnosis['loss_spikes'] = {
                    'count': len(spikes),
                    'max_spike': loss_diff.max(),
                    'spike_steps': df.loc[spikes.index, '_step'].tolist()[:5]  # First 5 spike steps
                }
        
        # Check gradient norms for instability
        if 'train/grad_norm' in df.columns:
            grad_norm = df['train/grad_norm'].dropna()
            if len(grad_norm) > 0:
                diagnosis['gradient_issues'] = {
                    'max_grad_norm': grad_norm.max(),
                    'exploding_gradients': len(grad_norm[grad_norm > 100]),
                    'grad_norm_spikes': len(grad_norm[grad_norm > grad_norm.mean() + 5 * grad_norm.std()])
                }
        
        # Check system metrics for OOM
        try:
            system_df = self.get_system_metrics()
            if not system_df.empty:
                gpu_mem_cols = [col for col in system_df.columns 
                               if 'gpu' in col.lower() and 'memory' in col.lower()]
                
                if gpu_mem_cols:
                    max_mem_values = []
                    for col in gpu_mem_cols:
                        if 'percent' in col.lower():
                            col_max = system_df[col].dropna().max()
                            if not np.isnan(col_max) and col_max <= 100:
                                max_mem_values.append(col_max)
                    
                    if max_mem_values:
                        diagnosis['max_gpu_memory_percent'] = max(max_mem_values)
                        diagnosis['potential_oom'] = max(max_mem_values) > 95
        except:
            pass
        
        # Get last logs for error messages
        last_logs = self.get_logs(lines=100)
        error_keywords = ['error', 'exception', 'traceback', 'cuda out of memory', 
                         'nan', 'inf', 'overflow', 'assert', 'killed', 'abort']
        
        errors_found = []
        for line in last_logs:
            line_lower = line.lower()
            for keyword in error_keywords:
                if keyword in line_lower:
                    errors_found.append(line.strip())
                    break
        
        diagnosis['error_lines'] = errors_found[-10:]  # Last 10 error lines
        diagnosis['has_errors'] = len(errors_found) > 0
        
        return diagnosis
    
    def get_logs(self, lines: int = 100) -> List[str]:
        """Get the last N lines of logs from the run."""
        try:
            files = self.run.files()
            log_files = [f for f in files if f.name.endswith('.log') or f.name == 'output.log']
            
            if not log_files:
                return ["No log files found for this run"]
            
            log_file = log_files[0]
            log_content = log_file.download(replace=True).read()
            
            if isinstance(log_content, bytes):
                log_content = log_content.decode('utf-8', errors='ignore')
            
            log_lines = log_content.split('\n')
            return log_lines[-lines:]
            
        except Exception as e:
            return [f"Error fetching logs: {str(e)}"]
    
    def plot_detailed_analysis(self, save_path: Optional[str] = None):
        """Create detailed plots for the analysis."""
        df = self.get_history()
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training Loss with trend
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train/lm_loss' in df.columns:
            loss_data = df[['_step', 'train/lm_loss']].dropna()
            ax1.plot(loss_data['_step'], loss_data['train/lm_loss'], label='LM Loss', alpha=0.7)
            
            # Add moving average
            if len(loss_data) > 100:
                ma = loss_data['train/lm_loss'].rolling(window=100).mean()
                ax1.plot(loss_data['_step'], ma, label='MA(100)', color='red', linewidth=2)
            
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Trend')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Plot 2: Learning Rate Schedule
        ax2 = fig.add_subplot(gs[0, 2])
        if 'train/learning_rate' in df.columns:
            lr_data = df[['_step', 'train/learning_rate']].dropna()
            ax2.plot(lr_data['_step'], lr_data['train/learning_rate'], color='orange')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # Plot 3: GA Loss Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        ga_plotted = False
        if 'train/ga_actual_loss' in df.columns:
            ga_data = df[['_step', 'train/ga_actual_loss', 'train/lm_loss']].dropna()
            if not ga_data.empty:
                ax3.scatter(ga_data['_step'], ga_data['train/ga_actual_loss'], 
                           label='GA Actual Loss', color='red', alpha=0.6, s=20)
                ax3.plot(ga_data['_step'], ga_data['train/lm_loss'], 
                        label='Normal Loss', color='blue', alpha=0.7)
                ga_plotted = True
        
        if ga_plotted:
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Loss')
            ax3.set_title('Gradient Ascent vs Normal Loss')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: GA Iterations
        ax4 = fig.add_subplot(gs[1, 2])
        if 'train/ga_iterations_this_step' in df.columns:
            ga_iter_data = df[['_step', 'train/ga_iterations_this_step']].dropna()
            if not ga_iter_data.empty:
                ax4.bar(ga_iter_data['_step'], ga_iter_data['train/ga_iterations_this_step'], 
                       width=1, color='darkred', alpha=0.7)
                ax4.set_xlabel('Step')
                ax4.set_ylabel('GA Iterations')
                ax4.set_title('GA Iterations per Step')
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Gradient Norms
        ax5 = fig.add_subplot(gs[2, :2])
        if 'train/grad_norm' in df.columns:
            grad_data = df[['_step', 'train/grad_norm']].dropna()
            ax5.plot(grad_data['_step'], grad_data['train/grad_norm'], 
                    label='Gradient Norm', color='green', alpha=0.7)
            ax5.set_xlabel('Step')
            ax5.set_ylabel('Gradient Norm')
            ax5.set_title('Gradient Norm Evolution')
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')
        
        # Plot 6: Training Speed
        ax6 = fig.add_subplot(gs[2, 2])
        if 'runtime/samples_per_sec' in df.columns:
            speed_data = df[['_step', 'runtime/samples_per_sec']].dropna()
            if not speed_data.empty:
                ax6.plot(speed_data['_step'], speed_data['runtime/samples_per_sec'], 
                        color='purple', alpha=0.7)
                ax6.set_xlabel('Step')
                ax6.set_ylabel('Samples/sec')
                ax6.set_title('Training Speed')
                ax6.grid(True, alpha=0.3)
        
        # Plot 7: Iteration Time Distribution
        ax7 = fig.add_subplot(gs[3, 0])
        if 'runtime/iteration_time' in df.columns:
            iter_times = df['runtime/iteration_time'].dropna()
            if len(iter_times) > 0:
                ax7.hist(iter_times, bins=50, alpha=0.7, color='teal', edgecolor='black')
                ax7.axvline(iter_times.mean(), color='red', linestyle='--', 
                           label=f'Mean: {iter_times.mean():.2f}s')
                ax7.set_xlabel('Iteration Time (s)')
                ax7.set_ylabel('Count')
                ax7.set_title('Iteration Time Distribution')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
        
        # Plot 8: Loss Distribution
        ax8 = fig.add_subplot(gs[3, 1])
        if 'train/lm_loss' in df.columns:
            losses = df['train/lm_loss'].dropna()
            if len(losses) > 0:
                ax8.hist(losses, bins=50, alpha=0.7, color='navy', edgecolor='black')
                ax8.axvline(losses.mean(), color='red', linestyle='--', 
                           label=f'Mean: {losses.mean():.4f}')
                ax8.set_xlabel('Loss')
                ax8.set_ylabel('Count')
                ax8.set_title('Loss Distribution')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
        
        # Plot 9: GPU Memory Usage
        ax9 = fig.add_subplot(gs[3, 2])
        try:
            system_df = self.get_system_metrics()
            if not system_df.empty:
                gpu_mem_cols = [col for col in system_df.columns 
                               if 'gpu' in col.lower() and 'memory' in col.lower() and 'percent' in col.lower()]
                if gpu_mem_cols:
                    for i, col in enumerate(gpu_mem_cols[:4]):  # Max 4 GPUs
                        mem_data = system_df[col].dropna()
                        if len(mem_data) > 0:
                            ax9.plot(range(len(mem_data)), mem_data, 
                                    label=f'GPU {i}', alpha=0.7)
                    ax9.set_xlabel('Time Step')
                    ax9.set_ylabel('Memory %')
                    ax9.set_title('GPU Memory Usage')
                    ax9.legend()
                    ax9.grid(True, alpha=0.3)
        except:
            pass
        
        plt.suptitle(f'Detailed Analysis: {self.run.name} ({self.run.id})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("="*100)
        report.append(f"W&B Run Analysis Report: eleutherai/AISI/9vbuhj4o")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*100)
        
        # 1. Basic Run Information
        info = self.get_run_info()
        report.append("\n## 1. RUN INFORMATION")
        report.append(f"- Name: {info['name']}")
        report.append(f"- ID: {info['id']}")
        report.append(f"- State: {info['state']}")
        report.append(f"- Runtime: {info['runtime']/3600:.2f} hours ({info['runtime']:.0f} seconds)")
        report.append(f"- Total Steps: {info['step']:,}")
        report.append(f"- URL: {info['url']}")
        
        # 2. Key Configuration Parameters
        report.append("\n## 2. KEY CONFIGURATION PARAMETERS")
        key_config = self.get_key_configuration()
        
        report.append("\n### Model Architecture:")
        for k, v in key_config['model'].items():
            report.append(f"  - {k}: {v}")
        
        report.append("\n### Training Configuration:")
        for k, v in key_config['training'].items():
            report.append(f"  - {k}: {v}")
        
        report.append("\n### Gradient Ascent Configuration:")
        for k, v in key_config['gradient_ascent'].items():
            report.append(f"  - {k}: {v}")
        
        report.append("\n### Parallelism Configuration:")
        for k, v in key_config['parallelism'].items():
            report.append(f"  - {k}: {v}")
        
        # 3. Training Metrics Analysis
        report.append("\n## 3. TRAINING METRICS ANALYSIS")
        train_metrics = self.analyze_training_metrics()
        
        if 'training_loss' in train_metrics:
            loss = train_metrics['training_loss']
            report.append(f"\n### Training Loss:")
            report.append(f"  - Initial Loss: {loss['initial']:.6f}" if loss['initial'] else "  - Initial Loss: N/A")
            report.append(f"  - Final Loss: {loss['final']:.6f}" if loss['final'] else "  - Final Loss: N/A")
            report.append(f"  - Minimum Loss: {loss['min']:.6f}")
            report.append(f"  - Mean ± Std: {loss['mean']:.6f} ± {loss['std']:.6f}")
            if 'improvement' in loss:
                report.append(f"  - Total Improvement: {loss['improvement']:.6f} ({loss['improvement_percent']:.2f}%)")
            if 'recent_trend' in loss:
                trend_direction = "improving" if loss['recent_trend'] < 0 else "worsening"
                report.append(f"  - Recent Trend: {trend_direction} ({loss['recent_trend']:.6f})")
        
        if 'learning_rate' in train_metrics:
            lr = train_metrics['learning_rate']
            report.append(f"\n### Learning Rate:")
            report.append(f"  - Initial: {lr['initial']:.2e}" if lr['initial'] else "  - Initial: N/A")
            report.append(f"  - Final: {lr['final']:.2e}" if lr['final'] else "  - Final: N/A")
            report.append(f"  - Range: [{lr['min']:.2e}, {lr['max']:.2e}]")
        
        # Gradient norms
        for metric_name in ['train/grad_norm', 'train/grad_norm_before_clip']:
            if metric_name in train_metrics:
                grad = train_metrics[metric_name]
                report.append(f"\n### {metric_name.replace('train/', '').replace('_', ' ').title()}:")
                report.append(f"  - Mean ± Std: {grad['mean']:.4f} ± {grad['std']:.4f}")
                report.append(f"  - Maximum: {grad['max']:.4f}")
                report.append(f"  - Number of Spikes (>3σ): {grad['spikes']}")
        
        # 4. Gradient Ascent Analysis
        ga_metrics = self.analyze_gradient_ascent_metrics()
        if ga_metrics and ga_metrics['config']['mode'] != 'none':
            report.append("\n## 4. GRADIENT ASCENT ANALYSIS")
            
            report.append(f"\n### GA Configuration:")
            for k, v in ga_metrics['config'].items():
                report.append(f"  - {k}: {v}")
            
            # GA loss metrics
            for metric_name in ['train/ga_actual_loss', 'train/ga_objective', 'train/ga_lr_scale']:
                if metric_name in ga_metrics:
                    values = ga_metrics[metric_name]
                    clean_name = metric_name.replace('train/', '').replace('_', ' ').title()
                    report.append(f"\n### {clean_name}:")
                    report.append(f"  - Mean ± Std: {values['mean']:.6f} ± {values['std']:.6f}")
                    report.append(f"  - Median [Q25, Q75]: {values['median']:.6f} [{values['q25']:.6f}, {values['q75']:.6f}]")
                    report.append(f"  - Range: [{values['min']:.6f}, {values['max']:.6f}]")
                    report.append(f"  - Number of GA Steps: {values['count']}")
                    if 'trend' in values:
                        trend_dir = "increasing" if values['trend'] > 0 else "decreasing"
                        report.append(f"  - Trend: {trend_dir} ({values['trend']:.6f})")
            
            # GA iterations
            for metric_name in ['train/ga_iterations_this_step', 'train/total_ga_iterations']:
                if metric_name in ga_metrics:
                    values = ga_metrics[metric_name]
                    clean_name = metric_name.replace('train/', '').replace('_', ' ').title()
                    report.append(f"\n### {clean_name}:")
                    report.append(f"  - Final Value: {values['final']}")
                    report.append(f"  - Maximum: {values['max']}")
                    if 'total_ga_steps' in values:
                        report.append(f"  - Total Steps with GA: {values['total_ga_steps']}")
            
            # GA efficiency
            if 'efficiency' in ga_metrics:
                eff = ga_metrics['efficiency']
                report.append(f"\n### GA Efficiency:")
                if eff['loss_increase_ratio']:
                    report.append(f"  - Loss Increase Ratio: {eff['loss_increase_ratio']:.2f}x")
                report.append(f"  - Average Loss Difference: {eff['loss_difference']:.6f}")
        
        # 5. Training Time Analysis
        report.append("\n## 5. TRAINING TIME ANALYSIS")
        time_metrics = self.analyze_training_time()
        
        report.append(f"  - Total Runtime: {time_metrics['total_runtime_hours']:.2f} hours")
        report.append(f"  - Total Steps: {time_metrics['total_steps']:,}")
        if 'avg_seconds_per_step' in time_metrics:
            report.append(f"  - Average Time per Step: {time_metrics['avg_seconds_per_step']:.2f} seconds")
            report.append(f"  - Steps per Hour: {time_metrics['steps_per_hour']:.1f}")
        
        if 'iteration_time' in time_metrics:
            iter_time = time_metrics['iteration_time']
            report.append(f"\n### Iteration Time Statistics:")
            report.append(f"  - Mean ± Std: {iter_time['mean']:.2f} ± {iter_time['std']:.2f} seconds")
            report.append(f"  - Range: [{iter_time['min']:.2f}, {iter_time['max']:.2f}] seconds")
            report.append(f"  - Number of Slowdowns (>2σ): {iter_time['slowdowns']}")
        
        if 'ga_overhead' in time_metrics:
            overhead = time_metrics['ga_overhead']
            report.append(f"\n### GA Time Overhead:")
            report.append(f"  - Average GA Step Time: {overhead['ga_step_time']:.2f} seconds")
            report.append(f"  - Average Normal Step Time: {overhead['normal_step_time']:.2f} seconds")
            if overhead['overhead_ratio']:
                report.append(f"  - Overhead Ratio: {overhead['overhead_ratio']:.2f}x")
        
        # Training speed
        if 'speed_metrics' in train_metrics and train_metrics['speed_metrics']:
            report.append(f"\n### Training Speed Metrics:")
            for metric, values in train_metrics['speed_metrics'].items():
                metric_name = metric.split('/')[-1].replace('_', ' ').title()
                report.append(f"  - {metric_name}: {values['mean']:.2f} ± {values['std']:.2f}")
        
        # 6. Crash/Issue Diagnosis
        report.append("\n## 6. CRASH/ISSUE DIAGNOSIS")
        diagnosis = self.diagnose_crash()
        
        report.append(f"  - Run State: {diagnosis['state']}")
        report.append(f"  - Exit Code: {diagnosis['exit_code']}")
        
        if 'loss_issues' in diagnosis:
            issues = diagnosis['loss_issues']
            report.append(f"\n### Loss Stability:")
            report.append(f"  - NaN Loss Count: {issues['nan_count']}")
            report.append(f"  - Inf Loss Count: {issues['inf_count']}")
            if issues['first_nan_step']:
                report.append(f"  - First NaN at Step: {issues['first_nan_step']}")
            report.append(f"  - Has Issues: {'Yes' if issues['has_issues'] else 'No'}")
        
        if 'loss_spikes' in diagnosis:
            spikes = diagnosis['loss_spikes']
            report.append(f"\n### Loss Spikes:")
            report.append(f"  - Number of Spikes: {spikes['count']}")
            report.append(f"  - Maximum Spike: {spikes['max_spike']:.6f}")
            if spikes['spike_steps']:
                report.append(f"  - Spike Steps: {spikes['spike_steps']}")
        
        if 'gradient_issues' in diagnosis:
            grad_issues = diagnosis['gradient_issues']
            report.append(f"\n### Gradient Stability:")
            report.append(f"  - Maximum Gradient Norm: {grad_issues['max_grad_norm']:.2f}")
            report.append(f"  - Exploding Gradients (>100): {grad_issues['exploding_gradients']}")
            report.append(f"  - Gradient Spikes (>5σ): {grad_issues['grad_norm_spikes']}")
        
        if 'max_gpu_memory_percent' in diagnosis:
            report.append(f"\n### GPU Memory:")
            report.append(f"  - Maximum GPU Memory Usage: {diagnosis['max_gpu_memory_percent']:.1f}%")
            if 'potential_oom' in diagnosis:
                report.append(f"  - Potential OOM Risk: {'Yes' if diagnosis['potential_oom'] else 'No'}")
        
        if diagnosis['has_errors']:
            report.append(f"\n### Errors Found in Logs:")
            for i, error in enumerate(diagnosis['error_lines'][-5:], 1):
                report.append(f"  {i}. {error[:200]}...")  # Truncate long error lines
        else:
            report.append(f"\n### No obvious errors found in logs")
        
        # 7. Summary and Recommendations
        report.append("\n## 7. SUMMARY AND RECOMMENDATIONS")
        
        # Determine overall health
        issues = []
        if diagnosis['state'] != 'finished':
            issues.append(f"Run did not finish successfully (state: {diagnosis['state']})")
        if 'loss_issues' in diagnosis and diagnosis['loss_issues']['has_issues']:
            issues.append("Loss instability detected (NaN/Inf values)")
        if 'gradient_issues' in diagnosis and diagnosis['gradient_issues']['exploding_gradients'] > 0:
            issues.append("Exploding gradients detected")
        if 'potential_oom' in diagnosis and diagnosis['potential_oom']:
            issues.append("High GPU memory usage - potential OOM risk")
        
        if issues:
            report.append("\n### Issues Detected:")
            for issue in issues:
                report.append(f"  - {issue}")
        else:
            report.append("\n### Run appears healthy with no major issues detected")
        
        # Performance insights
        report.append("\n### Performance Insights:")
        if 'training_loss' in train_metrics and 'improvement_percent' in train_metrics['training_loss']:
            imp = train_metrics['training_loss']['improvement_percent']
            report.append(f"  - Loss improved by {imp:.2f}% over the run")
        
        if ga_metrics and ga_metrics['config']['mode'] != 'none':
            report.append(f"  - Gradient Ascent mode: {ga_metrics['config']['mode']}")
            if 'efficiency' in ga_metrics and ga_metrics['efficiency']['loss_increase_ratio']:
                ratio = ga_metrics['efficiency']['loss_increase_ratio']
                report.append(f"  - GA achieves {ratio:.2f}x loss increase on average")
        
        report.append("\n" + "="*100)
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Analyze W&B run 9vbuhj4o')
    parser.add_argument('--api-key', type=str, default=None,
                       help='W&B API key (uses environment variable if not provided)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and display detailed plots')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to specified path')
    parser.add_argument('--logs', type=int, default=0,
                       help='Number of log lines to print (0 to skip)')
    parser.add_argument('--report', type=str, default='run_9vbuhj4o_analysis.txt',
                       help='Save analysis report to specified file (default: run_9vbuhj4o_analysis.txt)')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        print("Initializing analyzer for run eleutherai/AISI/9vbuhj4o...")
        analyzer = RunAnalyzer(args.api_key)
        
        # Generate and print report
        print("Generating comprehensive analysis report...")
        report = analyzer.generate_comprehensive_report()
        print(report)
        
        # Save report
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            print(f"\nReport saved to: {args.report}")
        
        # Show logs if requested
        if args.logs > 0:
            print(f"\n\n{'='*100}")
            print(f"Last {args.logs} log lines:")
            print('='*100)
            logs = analyzer.get_logs(args.logs)
            for line in logs:
                print(line)
        
        # Generate plots if requested
        if args.plot or args.save_plot:
            print("\nGenerating detailed analysis plots...")
            analyzer.plot_detailed_analysis(args.save_plot)
        
    except Exception as e:
        print(f"Error analyzing run: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())