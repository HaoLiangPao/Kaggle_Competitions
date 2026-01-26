"""
Inference Script for kaggle_s6_e1 Competition

Generate predictions on test data and create submission file.

Usage:
    python predict.py
    python predict.py --artifacts artifacts_dir --output my_submission.csv
"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path to import ml_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_pipeline import MLPipeline
from ml_pipeline.utils import load_config, setup_logging


def main():
    """Main inference function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate predictions for kaggle_s6_e1')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--artifacts', type=str, default=None,
                       help='Path to artifacts directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output submission file')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent / args.config
    config = load_config(str(config_path))
    
    # Setup logging
    log_config = config.get('logging', {})
    setup_logging(level=log_config.get('level', 'INFO'))
    
    # Paths
    script_dir = Path(__file__).parent
    artifacts_dir = args.artifacts or (script_dir / config['output']['artifacts_dir'])
    output_file = args.output or (script_dir / config['output']['submission_file'])
    test_path = script_dir / config['data']['test_path']
    sample_submission_path = script_dir / config['data']['sample_submission_path']
    
    print("="*60)
    print("Generating Predictions")
    print("="*60)
    print(f"Artifacts dir: {artifacts_dir}")
    print(f"Test data: {test_path}")
    print(f"Output file: {output_file}")
    print("="*60)
    
    # Initialize pipeline and load artifacts
    pipeline = MLPipeline(config)
    pipeline.load_artifacts(str(artifacts_dir))
    
    # Load and preprocess test data
    print("Loading and preprocessing test data...")
    pipeline.load_data(train_path=None, test_path=str(test_path))
    pipeline.X_test = pipeline.preprocessor.transform(pipeline.X_test)
    
    # Get test IDs for submission
    test_df = pd.read_csv(test_path)
    id_column = config['data']['id_column']
    
    # Generate predictions
    print("Generating predictions...")
    predictions = pipeline.predict()
    
    # Create submission file
    submission = pd.DataFrame({
        id_column: test_df[id_column],
        config['data']['target_column']: predictions
    })
    
    submission.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("Predictions generated successfully!")
    print(f"Submission file saved to: {output_file}")
    print(f"Prediction statistics:")
    print(f"  Mean: {predictions.mean():.2f}")
    print(f"  Std:  {predictions.std():.2f}")
    print(f"  Min:  {predictions.min():.2f}")
    print(f"  Max:  {predictions.max():.2f}")
    print("="*60)
    
    # Verify submission format
    sample_submission = pd.read_csv(sample_submission_path)
    if submission.shape == sample_submission.shape:
        print("\n✓ Submission format matches sample submission")
    else:
        print("\n⚠ Warning: Submission shape doesn't match sample submission")
        print(f"  Expected: {sample_submission.shape}")
        print(f"  Got: {submission.shape}")


if __name__ == '__main__':
    main()
