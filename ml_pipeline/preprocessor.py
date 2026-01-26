"""
Feature Preprocessing Module

Handles feature engineering, scaling, encoding, and missing value imputation.
Maintains fit/transform pattern for train/test consistency.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    Feature preprocessing with fit/transform pattern.
    
    Handles:
    - Numerical feature scaling and imputation
    - Categorical feature encoding
    - Feature engineering
    - Save/load preprocessing artifacts
    """
    
    def __init__(self,
                 numerical_features: Optional[List[str]] = None,
                 categorical_features: Optional[List[str]] = None,
                 scale_numerical: bool = True,
                 impute_strategy: str = 'median'):
        """
        Initialize FeaturePreprocessor.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            scale_numerical: Whether to scale numerical features
            impute_strategy: Strategy for missing value imputation ('mean', 'median', 'most_frequent')
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.scale_numerical = scale_numerical
        self.impute_strategy = impute_strategy
        
        # Initialize transformers
        self.num_imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler() if scale_numerical else None
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        self.is_fitted = False
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training feature DataFrame
            
        Returns:
            self
        """
        logger.info("Fitting preprocessor...")
        
        # Auto-detect feature types if not specified
        if not self.numerical_features and not self.categorical_features:
            self._auto_detect_feature_types(X)
        
        # Fit numerical transformers
        if self.numerical_features:
            X_num = X[self.numerical_features]
            self.num_imputer.fit(X_num)
            
            if self.scaler:
                X_num_imputed = self.num_imputer.transform(X_num)
                self.scaler.fit(X_num_imputed)
        
        # Fit categorical transformers
        if self.categorical_features:
            X_cat = X[self.categorical_features]
            self.cat_imputer.fit(X_cat)
            X_cat_imputed = self.cat_imputer.transform(X_cat)
            self.encoder.fit(X_cat_imputed)
        
        self.is_fitted = True
        self._build_feature_names()
        
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_out_)}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessor.
        
        Args:
            X: Feature DataFrame to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        transformed_parts = []
        
        # Transform numerical features
        if self.numerical_features:
            X_num = X[self.numerical_features]
            X_num_imputed = self.num_imputer.transform(X_num)
            
            if self.scaler:
                X_num_scaled = self.scaler.transform(X_num_imputed)
                transformed_parts.append(pd.DataFrame(
                    X_num_scaled, 
                    columns=self.numerical_features,
                    index=X.index
                ))
            else:
                transformed_parts.append(pd.DataFrame(
                    X_num_imputed,
                    columns=self.numerical_features,
                    index=X.index
                ))
        
        # Transform categorical features
        if self.categorical_features:
            X_cat = X[self.categorical_features]
            X_cat_imputed = self.cat_imputer.transform(X_cat)
            X_cat_encoded = self.encoder.transform(X_cat_imputed)
            
            cat_feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            transformed_parts.append(pd.DataFrame(
                X_cat_encoded,
                columns=cat_feature_names,
                index=X.index
            ))
        
        # Combine all transformed features
        X_transformed = pd.concat(transformed_parts, axis=1)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Training feature DataFrame
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X).transform(X)
    
    def _auto_detect_feature_types(self, X: pd.DataFrame):
        """
        Auto-detect numerical and categorical features.
        
        Args:
            X: Feature DataFrame
        """
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Auto-detected {len(self.numerical_features)} numerical features")
        logger.info(f"Auto-detected {len(self.categorical_features)} categorical features")
    
    def _build_feature_names(self):
        """Build output feature names after fitting."""
        feature_names = []
        
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        if self.categorical_features:
            cat_names = self.encoder.get_feature_names_out(self.categorical_features)
            feature_names.extend(cat_names)
        
        self.feature_names_out_ = feature_names
    
    def save(self, filepath: str):
        """
        Save preprocessor to file.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        joblib.dump(self, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FeaturePreprocessor':
        """
        Load preprocessor from file.
        
        Args:
            filepath: Path to saved file
            
        Returns:
            Loaded FeaturePreprocessor
        """
        preprocessor = joblib.load(filepath)
        logger.info(f"Preprocessor loaded from {filepath}")
        return preprocessor
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")
        return self.feature_names_out_
