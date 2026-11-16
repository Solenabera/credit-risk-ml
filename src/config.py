"""
Configuration Loader Module.
Loads system settings from YAML and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable override support.
    """
    if config_path is None:
        env_config = os.getenv("CONFIG_PATH")
        if env_config and os.path.exists(env_config):
            config_path = env_config
        else:
            config_path = str(get_project_root() / "config" / "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    if "MLFLOW_TRACKING_URI" in os.environ:
        config["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if "MLFLOW_EXPERIMENT_NAME" in os.environ:
        config["mlflow"]["experiment_name"] = os.environ["MLFLOW_EXPERIMENT_NAME"]
    if "MODEL_PATH" in os.environ:
        config["artifacts"]["pipeline_file"] = os.environ["MODEL_PATH"]
    if "OPTIMAL_THRESHOLD_PATH" in os.environ:
        config["artifacts"]["threshold_file"] = os.environ["OPTIMAL_THRESHOLD_PATH"]

    return config
