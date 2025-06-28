# W&B Run Analysis Tool for GPT-NeoX

This tool provides comprehensive analysis of Weights & Biases (W&B) runs for GPT-NeoX training experiments, with special support for gradient ascent metrics and crash diagnosis.

## Features

- **Training Metrics Analysis**: Analyze loss curves, learning rates, and training speed
- **Gradient Ascent Support**: Special handling for GA-specific metrics (ga_actual_loss, ga_objective, etc.)
- **Crash Diagnosis**: Identify potential causes of training failures (NaN losses, OOM, errors)
- **Log Inspection**: View recent logs to debug issues
- **Visualization**: Generate training curve plots
- **Report Generation**: Create comprehensive text reports
- **Batch Analysis**: Compare multiple runs or find crashed runs

## Installation

```bash
# Install dependencies
pip install -r requirements-analysis.txt

# Make script executable
chmod +x analyze_wandb_run.py
```

## Usage

### Basic Analysis

```bash
# Analyze a single run
python analyze_wandb_run.py entity/project/run_id

# Example with your run
python analyze_wandb_run.py eleutherai/AISI/ugcwac4b
```

### Advanced Options

```bash
# Show last 50 log lines
python analyze_wandb_run.py entity/project/run_id --logs 50

# Generate and save plots
python analyze_wandb_run.py entity/project/run_id --plot --save-plot training_curves.png

# Save analysis report to file
python analyze_wandb_run.py entity/project/run_id --report analysis_report.txt

# Use specific API key
python analyze_wandb_run.py entity/project/run_id --api-key YOUR_API_KEY
```

### Example Scripts

```bash
# Compare multiple runs
python analyze_wandb_examples.py --compare eleutherai/AISI run1 run2 run3

# Find crashed runs in a project
python analyze_wandb_examples.py --find-crashed eleutherai AISI

# Export all metrics to CSV
python analyze_wandb_examples.py --export eleutherai/AISI/run_id metrics.csv
```

## Key Metrics Analyzed

### Training Metrics
- **Loss**: Final, minimum, mean, and standard deviation
- **Learning Rate**: Initial and final values
- **Speed**: Samples/sec, tokens/sec, iteration time, FLOPS

### Gradient Ascent Metrics
- **GA Loss**: Actual loss and objective values during GA steps
- **GA Configuration**: Mode (interval/interleaved), iterations, intervals
- **GA Statistics**: Count, mean, std of GA metrics

### System Metrics
- **GPU Memory**: Maximum usage percentage
- **Runtime**: Total training time
- **Steps**: Total training iterations

### Crash Diagnosis
- **Exit Code**: Process exit status
- **NaN/Inf Detection**: Count of invalid loss values
- **Error Detection**: Scans logs for error keywords
- **GPU OOM**: Checks for out-of-memory conditions

## Output Examples

### Text Report
```
================================================================================
W&B Run Analysis Report
Generated: 2025-06-25 17:20:29
================================================================================

## Run Information
- Name: annealing_baseline_ga_interleaved_1_in_10_ga_lr_scale=0.5
- State: finished
- Runtime: 21.58 hours
- Total Steps: 11921

## Training Metrics
### Training Loss
- Final: 2.108042
- Min: 2.072538
- Mean ± Std: 2.187851 ± 0.088727

## Gradient Ascent Analysis
- Mode: interleaved
- GA Actual Loss Mean: 1.001903
- GA Objective Mean: -1.001903
```

### Programmatic Usage

```python
from analyze_wandb_run import WandbRunAnalyzer

# Initialize analyzer
analyzer = WandbRunAnalyzer("eleutherai/AISI/ugcwac4b")

# Get run info
info = analyzer.get_run_info()
print(f"Run state: {info['state']}")

# Analyze training metrics
metrics = analyzer.analyze_training_metrics()
final_loss = metrics['training_loss']['final']

# Check for gradient ascent
ga_metrics = analyzer.analyze_gradient_ascent_metrics()
if ga_metrics['config']['mode'] != 'none':
    print(f"GA mode: {ga_metrics['config']['mode']}")

# Diagnose potential issues
diagnosis = analyzer.diagnose_crash()
if diagnosis['error_lines']:
    print("Errors found in logs!")
```

## Troubleshooting

1. **Authentication Issues**: 
   - Run `wandb login` or set `WANDB_API_KEY` environment variable
   - Use `--api-key` flag to provide key directly

2. **Large Runs**:
   - The tool fetches up to 10,000 data points by default
   - For very long runs, consider using the CSV export feature

3. **Missing Metrics**:
   - Some metrics may not be available depending on the GPT-NeoX configuration
   - The tool gracefully handles missing metrics

## Notes

- The tool is specifically designed for GPT-NeoX training runs
- Gradient ascent metrics are only available when GA is enabled in training
- System metrics availability depends on W&B agent configuration