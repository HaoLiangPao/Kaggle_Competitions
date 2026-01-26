"""
Model Wrapper Module

Provides unified interface for different model types (LR, GBDT, etc.)
"""

import joblib
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")


class ModelWrapper:
    """
    Unified wrapper for different model types.
    
    Supports:
    - Linear Regression (lr)
    - Ridge Regression (ridge)
    - Lasso Regression (lasso)
    - Gradient Boosting (gbdt)
    - LightGBM (lightgbm)
    - XGBoost (xgboost)
    """
    
    SUPPORTED_MODELS = {
        'lr': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'gbdt': GradientBoostingRegressor,
    }
    
    if LIGHTGBM_AVAILABLE:
        SUPPORTED_MODELS['lightgbm'] = lgb.LGBMRegressor
    
    if XGBOOST_AVAILABLE:
        SUPPORTED_MODELS['xgboost'] = xgb.XGBRegressor
    
    def __init__(self, 
                 model_type: str = 'lr',
                 model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelWrapper.
        
        Args:
            model_type: Type of model ('lr', 'ridge', 'lasso', 'gbdt', 'lightgbm', 'xgboost')
            model_params: Parameters to pass to the model
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. "
                           f"Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.metadata = {
            'model_type': model_type,
            'model_params': model_params,
            'created_at': datetime.now().isoformat()
        }
        self.is_fitted = False
        
    def _create_model(self):
        """Create model instance based on type and parameters."""
        model_class = self.SUPPORTED_MODELS[self.model_type]
        return model_class(**self.model_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the model.
        
        Args:
            X: Training features
            y: Training target
            
        Returns:
            self
        """
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X, y)
        self.is_fitted = True
        self.metadata['fitted_at'] = datetime.now().isoformat()
        self.metadata['n_features'] = X.shape[1]
        self.metadata['n_samples'] = X.shape[0]
        logger.info("Model training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Model parameters dictionary
        """
        return self.model.get_params()
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
        """
        self.model.set_params(**params)
        self.model_params.update(params)
    
    def save(self, filepath: str, save_metadata: bool = True):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            save_metadata: Whether to save metadata alongside model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Save model
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save metadata
        if save_metadata:
            metadata_path = Path(filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"Metadata saved to {metadata_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelWrapper':
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded ModelWrapper instance
        """
        # Load metadata if available
        metadata_path = Path(filepath).with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            model_type = metadata.get('model_type', 'lr')
            model_params = metadata.get('model_params', {})
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
            model_type = 'lr'
            model_params = {}
        
        # Create wrapper instance
        wrapper = cls(model_type=model_type, model_params=model_params)
        
        # Load model
        wrapper.model = joblib.load(filepath)
        wrapper.is_fitted = True
        
        if metadata_path.exists():
            wrapper.metadata = metadata
        
        logger.info(f"Model loaded from {filepath}")
        return wrapper
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance Series or None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(self.model.feature_importances_)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(np.abs(self.model.coef_))
        else:
            logger.warning(f"{self.model_type} does not support feature importance")
            return None
