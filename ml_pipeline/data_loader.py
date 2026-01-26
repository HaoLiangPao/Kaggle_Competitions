"""
Data Loading and Validation Module

Handles data ingestion and basic validation checks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader with schema validation and basic checks.
    
    Features:
    - Load CSV/Parquet files
    - Schema validation
    - Data quality checks
    - Train/test splitting
    """
    
    def __init__(self, 
                 feature_columns: Optional[List[str]] = None,
                 target_column: Optional[str] = None,
                 id_column: Optional[str] = None,
                 validate_schema: bool = True):
        """
        Initialize DataLoader.
        
        Args:
            feature_columns: List of feature column names
            target_column: Name of target column
            id_column: Name of ID column to exclude from features
            validate_schema: Whether to validate schema on load
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.id_column = id_column
        self.validate_schema = validate_schema
        
    def load_csv(self, 
                 filepath: str, 
                 has_target: bool = True) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            has_target: Whether the data contains target column
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        if self.validate_schema:
            self._validate_schema(df, has_target)
            
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def load_parquet(self, 
                     filepath: str, 
                     has_target: bool = True) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            has_target: Whether the data contains target column
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {filepath}")
        df = pd.read_parquet(filepath)
        
        if self.validate_schema:
            self._validate_schema(df, has_target)
            
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _validate_schema(self, df: pd.DataFrame, has_target: bool = True):
        """
        Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate
            has_target: Whether target column should exist
            
        Raises:
            ValueError: If schema validation fails
        """
        # Check if feature columns exist
        if self.feature_columns:
            missing_features = set(self.feature_columns) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing feature columns: {missing_features}")
        
        # Check if target column exists
        if has_target and self.target_column:
            if self.target_column not in df.columns:
                raise ValueError(f"Missing target column: {self.target_column}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            logger.warning(f"Found completely empty columns: {empty_cols}")
        
        # Log data quality info
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        if missing_pct.max() > 0:
            logger.info(f"Missing value percentages:\n{missing_pct[missing_pct > 0]}")
    
    def get_features_and_target(self, 
                                df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if self.feature_columns:
            X = df[self.feature_columns]
        else:
            # Use all columns except target and id as features
            exclude_cols = [self.target_column, self.id_column]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols]
        
        y = df[self.target_column] if self.target_column in df.columns else None
        
        return X, y
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset statistics
        """
        info = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isna().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        return info
