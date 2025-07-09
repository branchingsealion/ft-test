"""
Logging utilities for the project.

This module provides functions for setting up and configuring logging.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(
    log_level: str = "INFO",
    log_file: str | Path | None = None,
    console: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Set up and configure the logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file (if None, no file logging)
        console: Whether to log to console
        log_format: Format string for log messages

    Returns:
        Configured logger
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Add file handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_dir = log_path.parent
        if str(log_dir) != "." and not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logger configured with level {log_level}")

    return root_logger

def log_memory_usage(logger: logging.Logger, step: str) -> dict[str, Any]:
    """
    Log memory usage statistics.

    Args:
        logger: Logger to use
        step: Current step description

    Returns:
        Dictionary with memory usage statistics
    """
    import psutil
    import torch

    # Get memory usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    ram_percent = process.memory_percent() / 100

    # Get GPU memory usage if available
    gpu_memory = 0
    gpu_utilization = 0
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
        gpu_utilization = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0

    # Create memory stats dictionary
    memory_stats = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "ram_gb": ram_usage,
        "ram_utilized": ram_percent,
        "gpu_memory_gb": gpu_memory,
        "gpu_utilized": gpu_utilization
    }

    # Log the memory usage
    logger.info(
        f"Memory usage - Step: {step}, "
        f"RAM: {ram_usage:.2f}GB ({ram_percent:.2%}), "
        f"GPU: {gpu_memory:.2f}GB ({gpu_utilization:.2%})"
    )

    return memory_stats

def create_log_entry(
    level: str,
    component: str,
    event: str,
    details: dict[str, Any] = None,
    memory_usage: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Create a structured log entry.

    Args:
        level: Log level (INFO, WARNING, ERROR, etc.)
        component: Component name (trainer, api, etc.)
        event: Event name (training_started, prediction_completed, etc.)
        details: Additional details
        memory_usage: Memory usage statistics

    Returns:
        Dictionary with log entry
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "component": component,
        "event": event,
        "details": details or {},
        "memory_usage": memory_usage or {}
    }

    return log_entry

def log_structured(logger: logging.Logger, entry: dict[str, Any]) -> None:
    """
    Log a structured entry.

    Args:
        logger: Logger to use
        entry: Log entry dictionary
    """
    import json

    # Convert to JSON string
    entry_json = json.dumps(entry)

    # Log at the appropriate level
    level = entry.get("level", "INFO").upper()
    numeric_level = getattr(logging, level, logging.INFO)
    logger.log(numeric_level, entry_json)

def configure_from_yaml(config_path: str | Path) -> logging.Logger:
    """
    Configure logging from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Configured logger
    """
    import yaml

    # Load configuration
    config_path = Path(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract logging configuration
    logging_config = config.get("logging", {})

    # Set up logger
    log_level = logging_config.get("level", "INFO")
    log_file = logging_config.get("file", None)
    console = logging_config.get("console", True)
    log_format = logging_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    return setup_logger(log_level, log_file, console, log_format)
