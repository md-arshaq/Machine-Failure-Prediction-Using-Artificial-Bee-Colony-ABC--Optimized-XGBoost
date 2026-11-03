"""
Utility functions for logging, seed setting, file I/O, and timing.
"""

import json
import logging
import os
import random
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import numpy as np
import pandas as pd
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across random and NumPy for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logger(
    name: str = "abc_xgboost",
    log_file: Optional[str] = "results/logs/experiment.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a standardized logger with console and file output.

    Args:
        name: Name of the logger.
        log_file: Path to destination log file.
        level: Logging verbosity level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary of parsed configuration values.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """
    Save serializable data to a JSON file using atomic writing.

    Args:
        data: Python object to serialize.
        path: Output file path.
        indent: Indentation spaces for formatting.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Atomic write via temporary file
    temp_dir = dir_name if dir_name else None
    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        json.dump(data, tf, indent=indent, default=str)
        temp_name = tf.name

    os.replace(temp_name, path)


def load_json(path: str) -> Any:
    """
    Load data from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed Python object.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_markdown_table(df: pd.DataFrame, path: str, title: str = "") -> None:
    """
    Save a pandas DataFrame as a GitHub-flavored Markdown table.

    Args:
        df: Pandas DataFrame to export.
        path: Destination file path (.md).
        title: Optional title header for the table.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    content = ""
    if title:
        content += f"### {title}\n\n"
    content += df.to_markdown(index=False) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@contextmanager
def timer(description: str = "Operation") -> Generator[Dict[str, float], None, None]:
    """
    Context manager to accurately measure execution time in seconds.

    Args:
        description: Text description of the timed block.

    Yields:
        Dictionary containing the elapsed time in 'elapsed'.
    """
    timing_info: Dict[str, float] = {}
    start_time = time.perf_counter()
    try:
        yield timing_info
    finally:
        elapsed = time.perf_counter() - start_time
        timing_info["elapsed"] = elapsed
        logging.getLogger("abc_xgboost").info(f"{description} completed in {elapsed:.3f} seconds.")

