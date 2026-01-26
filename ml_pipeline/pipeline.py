"""
ML Pipeline Orchestrator

End-to-end pipeline for training and inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import logging

from .data_loader import DataLoader
from .preprocessor import FeaturePreprocessor
from .model_wrapper import ModelWrapper
from .evaluator import Evaluator
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class MLPipeline:
    """
    End-to-end ML pipeline orchestrator.
    
    Manages:
    - Data loading and validation
    - Feature preprocessing
    - Model training
    - Evaluation
    - Artifact saving
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - data: Data paths and column definitions
                - preprocessing: Preprocessing settings
                - model: Model type and parameters
                - training: Training settings (CV folds, etc.)
                - output: Output paths for artifacts
        """
        self.config = config
        
        # Initialize components
        self.data_loader = self._init_data_loader()
        self.preprocessor = self._init_preprocessor()
        self.model = self._init_model()
        self.evaluator = self._init_evaluator()
        
        # Storage for training artifacts
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.cv_results = None
        
    def _init_data_loader(self) -> DataLoader:
        """Initialize DataLoader from config."""
        data_config = self.config.get('data', {})
        return DataLoader(
            feature_columns=data_config.get('feature_columns'),
            target_column=data_config.get('target_column'),
            id_column=data_config.get('id_column'),
            validate_schema=data_config.get('validate_schema', True)
        )
    
    def _init_preprocessor(self) -> FeaturePreprocessor:
        """Initialize FeaturePreprocessor from config."""
        prep_config = self.config.get('preprocessing', {})
        return FeaturePreprocessor(
            numerical_features=prep_config.get('numerical_features'),
            categorical_features=prep_config.get('categorical_features'),
            scale_numerical=prep_config.get('scale_numerical', True),
            impute_strategy=prep_config.get('impute_strategy', 'median')
        )
    
    def _init_model(self) -> ModelWrapper:
        """Initialize ModelWrapper from config."""
        model_config = self.config.get('model', {})
        return ModelWrapper(
            model_type=model_config.get('type', 'lr'),
            model_params=model_config.get('params', {})
        )
    
    def _init_evaluator(self) -> Evaluator:
        """Initialize Evaluator from config."""
        eval_config = self.config.get('evaluation', {})
        return Evaluator(metrics=eval_config.get('metrics'))
    
    def load_data(self, train_path: Optional[str] = None, 
                  test_path: Optional[str] = None):
        """
        Load training and test data.
        
        Args:
            train_path: Path to training data (overrides config)
            test_path: Path to test data (overrides config)
        """
        data_config = self.config.get('data', {})
        
        # Use provided paths or fall back to config
        train_path = train_path or data_config.get('train_path')
        test_path = test_path or data_config.get('test_path')
        
        # Load training data
        if train_path:
            logger.info("Loading training data...")
            self.train_data = self.data_loader.load_csv(train_path, has_target=True)
            self.X_train, self.y_train = self.data_loader.get_features_and_target(self.train_data)
        
        # Load test data
        if test_path:
            logger.info("Loading test data...")
            self.test_data = self.data_loader.load_csv(test_path, has_target=False)
            self.X_test, _ = self.data_loader.get_features_and_target(self.test_data)
    
    def preprocess(self):
        """Fit preprocessor on training data and transform train/test."""
        logger.info("Preprocessing features...")
        
        # Fit on training data
        self.X_train = self.preprocessor.fit_transform(self.X_train)
        
        # Transform test data if available
        if self.X_test is not None:
            self.X_test = self.preprocessor.transform(self.X_test)
        
        logger.info(f"Preprocessed features shape: {self.X_train.shape}")
    
    def train(self):
        """Train model on training data."""
        logger.info("Training model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Evaluate on training data
        y_train_pred = self.model.predict(self.X_train)
        self.evaluator.print_evaluation(self.y_train, y_train_pred, prefix="Training")
    
    def cross_validate(self, cv_folds: Optional[int] = None):
        """
        Perform cross-validation.
        
        Args:
            cv_folds: Number of CV folds (overrides config)
        """
        training_config = self.config.get('training', {})
        cv_folds = cv_folds or training_config.get('cv_folds', 5)
        
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        # Use detailed CV for comprehensive metrics
        self.cv_results = self.evaluator.cross_validate_detailed(
            self.model.model,  # Use underlying sklearn model
            self.X_train,
            self.y_train,
            cv=cv_folds
        )
        
        return self.cv_results
    
    def predict(self, X: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions on test data or provided data.
        
        Args:
            X: Features to predict on (uses self.X_test if None)
            
        Returns:
            Predictions array
        """
        if X is None:
            if self.X_test is None:
                raise ValueError("No test data available for prediction")
            X = self.X_test
        
        return self.model.predict(X)
    
    def save_artifacts(self, output_dir: Optional[str] = None):
        """
        Save all pipeline artifacts.
        
        Args:
            output_dir: Output directory (overrides config)
        """
        output_config = self.config.get('output', {})
        output_dir = output_dir or output_config.get('artifacts_dir', './artifacts')
        
        # Ensure output directory exists
        output_path = ensure_dir(output_dir)
        logger.info(f"Saving artifacts to {output_path}")
        
        # Save model
        model_path = output_path / 'model.pkl'
        self.model.save(str(model_path))
        
        # Save preprocessor
        preprocessor_path = output_path / 'preprocessor.pkl'
        self.preprocessor.save(str(preprocessor_path))
        
        # Save CV results if available
        if self.cv_results is not None:
            cv_results_path = output_path / 'cv_results.csv'
            self.cv_results.to_csv(cv_results_path, index=False)
            logger.info(f"CV results saved to {cv_results_path}")
            
            # Save CV summary
            cv_summary = {
                'mean_metrics': self.cv_results.drop('fold', axis=1).mean().to_dict(),
                'std_metrics': self.cv_results.drop('fold', axis=1).std().to_dict()
            }
            cv_summary_path = output_path / 'cv_summary.json'
            save_json(cv_summary, str(cv_summary_path))
        
        logger.info("All artifacts saved successfully")
    
    def load_artifacts(self, artifacts_dir: str):
        """
        Load saved artifacts for inference.
        
        Args:
            artifacts_dir: Directory containing saved artifacts
        """
        artifacts_path = Path(artifacts_dir)
        
        # Load model
        model_path = artifacts_path / 'model.pkl'
        self.model = ModelWrapper.load(str(model_path))
        
        # Load preprocessor
        preprocessor_path = artifacts_path / 'preprocessor.pkl'
        self.preprocessor = FeaturePreprocessor.load(str(preprocessor_path))
        
        logger.info(f"Artifacts loaded from {artifacts_dir}")
    
    def run_full_pipeline(self, train_path: str, test_path: Optional[str] = None):
        """
        Run complete training pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Optional path to test data
        """
        logger.info("="*60)
        logger.info("Starting Full ML Pipeline")
        logger.info("="*60)
        
        # Load data
        self.load_data(train_path, test_path)
        
        # Preprocess
        self.preprocess()
        
        # Cross-validate
        self.cross_validate()
        
        # Train final model on full training data
        self.train()
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("="*60)
        logger.info("Pipeline Completed Successfully")
        logger.info("="*60)
