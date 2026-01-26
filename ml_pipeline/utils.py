"""
Utility Functions

Helper functions for logging, path management, and configuration.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any
import logging


def setup_logging(level: str = 'INFO', log_file: str = None):
    """
    Setup logging configuration.
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to JSON file
        indent: JSON indentation
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data
