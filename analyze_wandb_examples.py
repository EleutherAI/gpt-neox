#!/usr/bin/env python3
"""
Example usage of the W&B run analyzer for different scenarios.
"""

from analyze_wandb_run import WandbRunAnalyzer
import argparse


def analyze_multiple_runs(entity, project, run_ids):
    """Analyze multiple runs and compare their performance."""
    results = {}
    
    for run_id in run_ids:
        run_path = f"{entity}/{project}/{run_id}"
        try:
            analyzer = WandbRunAnalyzer(run_path)
            
            # Get key metrics
            info = analyzer.get_run_info()
            train_metrics = analyzer.analyze_training_metrics()
            ga_metrics = analyzer.analyze_gradient_ascent_metrics()
            
            results[run_id] = {
                'name': info['name'],
                'state': info['state'],
                'runtime_hours': info['runtime'] / 3600,
                'final_loss': train_metrics.get('training_loss', {}).get('final', None),
                'min_loss': train_metrics.get('training_loss', {}).get('min', None),
                'ga_mode': ga_metrics['config']['mode']
            }
            
        except Exception as e:
            print(f"Error analyzing {run_id}: {e}")
            continue
    
    # Print comparison table
    print("\nRun Comparison:")
    print("-" * 100)
    print(f"{'Run ID':<15} {'State':<10} {'Runtime (h)':<12} {'Final Loss':<12} {'Min Loss':<12} {'GA Mode':<10}")
    print("-" * 100)
    
    for run_id, metrics in results.items():
        print(f"{run_id:<15} {metrics['state']:<10} {metrics['runtime_hours']:<12.2f} "
              f"{metrics['final_loss'] or 'N/A':<12} {metrics['min_loss'] or 'N/A':<12} "
              f"{metrics['ga_mode']:<10}")


def find_crashed_runs(entity, project, max_runs=10):
    """Find and analyze crashed runs in a project."""
    from wandb.apis.public import Api
    api = Api()
    
    # Get crashed/failed runs
    runs = api.runs(f"{entity}/{project}", 
                   filters={"state": {"$in": ["crashed", "failed"]}})
    
    crashed_info = []
    
    for i, run in enumerate(runs):
        if i >= max_runs:
            break
            
        run_path = f"{entity}/{project}/{run.id}"
        analyzer = WandbRunAnalyzer(run_path)
        
        diagnosis = analyzer.diagnose_crash()
        
        crashed_info.append({
            'id': run.id,
            'name': run.name,
            'exit_code': diagnosis['exit_code'],
            'runtime': run.summary.get('_runtime', 0) / 3600,
            'last_step': diagnosis['last_step'],
            'has_nan': diagnosis.get('loss_issues', {}).get('has_issues', False),
            'errors': len(diagnosis.get('error_lines', []))
        })
    
    # Print crash analysis
    print(f"\nCrashed Runs Analysis for {entity}/{project}:")
    print("-" * 80)
    
    for info in crashed_info:
        print(f"\nRun: {info['name']} ({info['id']})")
        print(f"  - Runtime: {info['runtime']:.2f} hours")
        print(f"  - Last step: {info['last_step']}")
        print(f"  - Exit code: {info['exit_code']}")
        print(f"  - Has NaN losses: {info['has_nan']}")
        print(f"  - Error lines found: {info['errors']}")


def export_metrics_to_csv(run_path, output_file):
    """Export all metrics from a run to CSV for further analysis."""
    import pandas as pd
    
    analyzer = WandbRunAnalyzer(run_path)
    
    # Get full history
    df = analyzer.get_history()
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} rows of metrics to {output_file}")
    
    # Print summary of available metrics
    print("\nAvailable metrics in the export:")
    for col in sorted(df.columns):
        non_null = df[col].notna().sum()
        if non_null > 0:
            print(f"  - {col}: {non_null} values")


def main():
    parser = argparse.ArgumentParser(description='Examples of W&B run analysis')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple runs (format: entity/project run_id1 run_id2 ...)')
    parser.add_argument('--find-crashed', nargs=2,
                       help='Find crashed runs (format: entity project)')
    parser.add_argument('--export', nargs=2,
                       help='Export metrics to CSV (format: entity/project/run_id output.csv)')
    
    args = parser.parse_args()
    
    if args.compare and len(args.compare) >= 3:
        entity_project = args.compare[0]
        entity, project = entity_project.split('/')
        run_ids = args.compare[1:]
        analyze_multiple_runs(entity, project, run_ids)
    
    elif args.find_crashed:
        entity, project = args.find_crashed
        find_crashed_runs(entity, project)
    
    elif args.export:
        run_path, output_file = args.export
        export_metrics_to_csv(run_path, output_file)
    
    else:
        print("Usage examples:")
        print("  Compare runs: python analyze_wandb_examples.py --compare entity/project run1 run2 run3")
        print("  Find crashed: python analyze_wandb_examples.py --find-crashed entity project")
        print("  Export to CSV: python analyze_wandb_examples.py --export entity/project/run_id output.csv")


if __name__ == "__main__":
    main()