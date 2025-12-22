"""
Configuration management utilities.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Manage application configuration."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load environment variables
        load_dotenv()

        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def get_env(self, key: str, default: str = None) -> str:
        """Get environment variable."""
        return os.getenv(key, default)

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return Path(self.get("data_dir", "./data"))

    @property
    def model_dir(self) -> Path:
        """Get model directory path."""
        return Path(self.get("model_dir", "./data/models"))

    @property
    def device(self) -> str:
        """Get compute device (cpu/cuda)."""
        return self.get("device", "cuda" if self.get_env("CUDA_VISIBLE_DEVICES") else "cpu")
