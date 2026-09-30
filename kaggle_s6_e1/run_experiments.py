"""
Experiment Runner Script

Orchestrates training and comparison of multiple models defined in data configurations.
"""

import subprocess
import pandas as pd
import json
from pathlib import Path
import sys

# Define configs to run
CONFIGS = [
    "config_lr.yaml",
    "config_ridge.yaml",
    "config_gbdt.yaml"
]

def run_experiment(config_file):
    """Run training and prediction for a given config."""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config_file}")
    print(f"{'='*60}")
    
    # Run training
    cmd_train = [sys.executable, "train.py", "--config", config_file]
    print(f"Executing: {' '.join(cmd_train)}")
    result_train = subprocess.run(cmd_train, capture_output=False)
    
    if result_train.returncode != 0:
        print(f"Error training {config_file}")
        return False
        
    # Run prediction (to generate submission file)
    cmd_predict = [sys.executable, "predict.py", "--config", config_file]
    print(f"Executing: {' '.join(cmd_predict)}")
    result_predict = subprocess.run(cmd_predict, capture_output=False)
    
    if result_predict.returncode != 0:
        print(f"Error predicting {config_file}")
        return False
        
    return True

def parse_results(config_file):
    """Parse results from artifacts."""
    # We need to read the config to know where artifacts are
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    artifacts_dir = config['output']['artifacts_dir']
    cv_summary_path = Path(artifacts_dir) / 'cv_summary.json'
    
    if not cv_summary_path.exists():
        return None
        
    with open(cv_summary_path, 'r') as f:
        summary = json.load(f)
        
    submission_path = config['output']['submission_file']
    submission_df = pd.read_csv(submission_path)
    
    return {
        'model': config['model']['type'],
        'cv_rmse': summary['mean_metrics']['rmse'],
        'cv_mae': summary['mean_metrics']['mae'],
        'cv_r2': summary['mean_metrics']['r2'],
        'pred_mean': submission_df[config['data']['target_column']].mean(),
        'pred_std': submission_df[config['data']['target_column']].std()
    }

def main():
    print("Starting Multi-Model Experiment Run...")
    
    results = []
    
    for config_file in CONFIGS:
        success = run_experiment(config_file)
        if success:
            res = parse_results(config_file)
            if res:
                res['config'] = config_file
                results.append(res)
    
    # Compare Results
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    
    # formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    cols = ['config', 'model', 'cv_rmse', 'cv_mae', 'cv_r2', 'pred_mean', 'pred_std']
    print(results_df[cols].to_string(index=False))
    
    print("\nBest model by CV RMSE:")
    best_model = results_df.loc[results_df['cv_rmse'].idxmin()]
    print(f"{best_model['config']} ({best_model['model']}) with RMSE: {best_model['cv_rmse']:.4f}")
    
    print("="*80)

if __name__ == "__main__":
    main()
