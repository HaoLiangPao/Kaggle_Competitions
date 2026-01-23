"""
Training Script for kaggle_s6_e1 Competition

Run this script to train the Linear Regression model with cross-validation.

Usage:
    python train.py
    python train.py --config custom_config.yaml
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import ml_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline import MLPipeline
from ml_pipeline.utils import load_config, setup_logging


def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model for kaggle_s6_e1')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    
    # Setup logging
    log_config = config.get('logging', {})
    log_file = Path(__file__).parent / log_config.get('log_file', 'training.log')
    setup_logging(
        level=log_config.get('level', 'INFO'),
        log_file=str(log_file)
    )
    
    # Initialize pipeline
    pipeline = MLPipeline(config)
    
    # Data paths (relative to script directory)
    script_dir = Path(__file__).parent
    train_path = script_dir / config['data']['train_path']
    test_path = script_dir / config['data']['test_path']
    
    # Run full training pipeline
    pipeline.run_full_pipeline(
        train_path=str(train_path),
        test_path=str(test_path)
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"Artifacts saved to: {script_dir / config['output']['artifacts_dir']}")
    print("="*60)
    
    # Print CV results summary
    if pipeline.cv_results is not None:
        print("\nCross-Validation Summary:")
        print("-" * 60)
        cv_summary = pipeline.cv_results.drop('fold', axis=1)
        print(cv_summary.describe().loc[['mean', 'std']])
        print("-" * 60)


if __name__ == '__main__':
    main()
