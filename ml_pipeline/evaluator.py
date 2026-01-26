"""
Model Evaluation Module

Provides metrics computation and cross-validation functionality.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Model evaluation with various regression metrics.
    
    Supports:
    - RMSE, MAE, R², MAPE
    - Cross-validation
    - Custom metrics
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize Evaluator.
        
        Args:
            metrics: List of metrics to compute ('rmse', 'mae', 'r2', 'mape')
                    If None, computes all available metrics
        """
        self.available_metrics = {
            'rmse': self._rmse,
            'mae': self._mae,
            'r2': self._r2,
            'mape': self._mape
        }
        
        if metrics is None:
            self.metrics = list(self.available_metrics.keys())
        else:
            # Validate metrics
            invalid = set(metrics) - set(self.available_metrics.keys())
            if invalid:
                raise ValueError(f"Invalid metrics: {invalid}. "
                               f"Choose from: {list(self.available_metrics.keys())}")
            self.metrics = metrics
    
    def evaluate(self, 
                 y_true: np.ndarray, 
                 y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all specified metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metric names and values
        """
        results = {}
        
        for metric_name in self.metrics:
            metric_func = self.available_metrics[metric_name]
            results[metric_name] = metric_func(y_true, y_pred)
        
        return results
    
    def cross_validate(self,
                      model,
                      X: pd.DataFrame,
                      y: pd.Series,
                      cv: int = 5,
                      scoring: str = 'neg_mean_squared_error') -> Dict[str, any]:
        """
        Perform cross-validation.
        
        Args:
            model: Model object with fit/predict methods
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric for sklearn
            
        Returns:
            Dictionary with CV scores and statistics
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        
        # Convert to positive RMSE if using MSE
        if scoring == 'neg_mean_squared_error':
            scores = np.sqrt(-scores)
            metric_name = 'RMSE'
        elif scoring.startswith('neg_'):
            scores = -scores
            metric_name = scoring[4:].upper()
        else:
            metric_name = scoring.upper()
        
        results = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'metric': metric_name
        }
        
        logger.info(f"CV {metric_name}: {results['mean']:.4f} (+/- {results['std']:.4f})")
        
        return results
    
    def cross_validate_detailed(self,
                               model,
                               X: pd.DataFrame,
                               y: pd.Series,
                               cv: int = 5) -> pd.DataFrame:
        """
        Perform detailed cross-validation with per-fold metrics.
        
        Args:
            model: Model object
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            DataFrame with metrics for each fold
        """
        logger.info(f"Performing detailed {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train and predict
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            # Compute metrics
            fold_metrics = self.evaluate(y_val_fold, y_pred)
            fold_metrics['fold'] = fold_idx
            fold_results.append(fold_metrics)
        
        results_df = pd.DataFrame(fold_results)
        
        # Log summary
        logger.info("\nCross-Validation Summary:")
        for metric in self.metrics:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            logger.info(f"{metric.upper()}: {mean_val:.4f} (+/- {std_val:.4f})")
        
        return results_df
    
    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate RMSE."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAE."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R²."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate MAPE."""
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return np.inf
        return mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    
    def print_evaluation(self, 
                        y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        prefix: str = ""):
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for logging (e.g., "Train", "Test")
        """
        results = self.evaluate(y_true, y_pred)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"{prefix} Evaluation Results:")
        logger.info(f"{'='*50}")
        
        for metric_name, value in results.items():
            logger.info(f"{metric_name.upper():>10s}: {value:.4f}")
        
        logger.info(f"{'='*50}\n")
