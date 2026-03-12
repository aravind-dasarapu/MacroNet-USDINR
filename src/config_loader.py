"""
Configuration loader for FX prediction pipeline.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configuration manager for the FX prediction system."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            base_path = Path(__file__).resolve().parent
            config_path = base_path / ".."/"config"/"config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def get(self, key: str, default=None):
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str):
        """Allow dictionary-style access."""
        return self.get(key)
    
    @property
    def data(self):
        return self._config['data']
    
    @property
    def preprocessing(self):
        return self._config['preprocessing']
    
    @property
    def model(self):
        return self._config['model']
    
    @property
    def training(self):
        return self._config['training']
    
    @property
    def monitoring(self):
        return self._config['monitoring']
    
    @property
    def deployment(self):
        return self._config['deployment']


def get_config(config_path: str = None) -> Config:
    """Factory function to get configuration instance."""
    return Config(config_path)