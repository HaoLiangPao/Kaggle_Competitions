"""
Reusable ML Pipeline Package

A modular, production-ready machine learning pipeline for Kaggle competitions
and real-world ML projects.

Components:
- DataLoader: Data loading and validation
- FeaturePreprocessor: Feature engineering and preprocessing
- ModelWrapper: Unified model interface
- Evaluator: Model evaluation and metrics
- MLPipeline: End-to-end orchestrator
"""

from .data_loader import DataLoader
from .preprocessor import FeaturePreprocessor
from .model_wrapper import ModelWrapper
from .evaluator import Evaluator
from .pipeline import MLPipeline

__version__ = '0.1.0'

__all__ = [
    'DataLoader',
    'FeaturePreprocessor',
    'ModelWrapper',
    'Evaluator',
    'MLPipeline',
]
