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

from ml_pipeline import DataLoader, FeaturePreprocessor, ModelWrapper, Evaluator
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
    
    # Initialize pipeline components explicitly for flexibility
    # This allows you to swap components or add custom steps easily
    
    # 1. Data Loading
    print("--------------------------------------------------")
    print("1. Data Loading")
    data_config = config.get('data', {})
    data_loader = DataLoader(
        feature_columns=data_config.get('feature_columns'),
        target_column=data_config.get('target_column'),
        id_column=data_config.get('id_column'),
        validate_schema=data_config.get('validate_schema', True)
    )
    
    script_dir = Path(__file__).parent
    train_path = script_dir / data_config['train_path']
    test_path = script_dir / data_config['test_path']
    
    # Load data
    train_df = data_loader.load_csv(str(train_path), has_target=True)
    test_df = data_loader.load_csv(str(test_path), has_target=False)
    
    # Split into X and y
    X_train, y_train = data_loader.get_features_and_target(train_df)
    X_test, _ = data_loader.get_features_and_target(test_df)
    
    # 2. Preprocessing
    print("\n--------------------------------------------------")
    print("2. Preprocessing")
    prep_config = config.get('preprocessing', {})
    preprocessor = FeaturePreprocessor(
        numerical_features=prep_config.get('numerical_features'),
        categorical_features=prep_config.get('categorical_features'),
        scale_numerical=prep_config.get('scale_numerical', True),
        impute_strategy=prep_config.get('impute_strategy', 'median')
    )
    
    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Processed shape: {X_train_processed.shape}")
    
    # 3. Model Initialization
    print("\n--------------------------------------------------")
    print("3. Model Initialization")
    model_config = config.get('model', {})
    model = ModelWrapper(
        model_type=model_config.get('type', 'lr'),
        model_params=model_config.get('params', {})
    )
    
    # 4. Cross Validation (Verification)
    print("\n--------------------------------------------------")
    print("4. Cross Validation")
    eval_config = config.get('evaluation', {})
    evaluator = Evaluator(metrics=eval_config.get('metrics'))
    train_config = config.get('training', {})
    
    cv_results = evaluator.cross_validate_detailed(
        model.model,  # Pass the underlying sklearn model
        X_train_processed, 
        y_train, 
        cv=train_config.get('cv_folds', 5)
    )
    
    # Print CV Summary
    print("\nCross-Validation Summary:")
    print(cv_results.mean(numeric_only=True))

    # 5. Final Training
    print("\n--------------------------------------------------")
    print("5. Final Training")
    model.fit(X_train_processed, y_train)
    
    # Verify on training set
    y_pred_train = model.predict(X_train_processed)
    evaluator.print_evaluation(y_train, y_pred_train, prefix="Final Training")

    # 6. Save Artifacts
    print("\n--------------------------------------------------")
    print("6. Saving Artifacts")
    output_config = config.get('output', {})
    artifacts_dir = script_dir / output_config.get('artifacts_dir', 'artifacts')
    output_path = Path(artifacts_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save components
    model.save(str(output_path / 'model.pkl'))
    preprocessor.save(str(output_path / 'preprocessor.pkl'))
    cv_results.to_csv(output_path / 'cv_results.csv', index=False)
    
    # Save CV summary for experiment runner
    cv_summary = {
        'mean_metrics': cv_results.drop('fold', axis=1).mean().to_dict(),
        'std_metrics': cv_results.drop('fold', axis=1).std().to_dict()
    }
    import json
    with open(output_path / 'cv_summary.json', 'w') as f:
        json.dump(cv_summary, f, indent=2)
    
    print(f"Artifacts saved to: {artifacts_dir}")
    print("--------------------------------------------------")


if __name__ == '__main__':
    main()
