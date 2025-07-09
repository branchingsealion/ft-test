#!/usr/bin/env python
"""
Training script for medical code prediction models with DuckDB support.

This script is a modified version of the original train.py script that adds support
for loading data from a DuckDB database. It uses the loader_duckdb.py module when
the use_duckdb parameter is set to true in the configuration.
"""

import argparse
import logging
import sys
import os
import yaml
from pathlib import Path
from typing import Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parents[2].absolute()))

# Import project modules
from src.model.trainer import create_trainer_from_config
from src.utils.logging_utils import setup_logger
from src.data.loader_tokenized import load_tokenized_from_duckdb

# Configure logger
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Train a medical code prediction model with DuckDB support"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training-config.yaml",
        help="Path to the training configuration file"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the trained model (overrides config)"
    )

    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the DuckDB database file (overrides config)"
    )

    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["mimic-iii", "mimic-iv"],
        help="Version of the MIMIC dataset (overrides config)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to run training on (cuda or cpu, overrides config)"
    )

    parser.add_argument(
        "--save-checkpoint-epochs",
        type=int,
        help="Number of epochs after which to save checkpoints (overrides config)"
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        help="Path to a checkpoint to resume training from"
    )

    parser.add_argument(
        "--tokens-dir",
        type=str,
        help="Path to the directory containing tokenized datasets (overrides config)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )

    return parser.parse_args()


def load_data(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Load tokenized training data from DuckDB.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dictionary containing the training, validation, and test datasets
    """
    data_config = config.get("data", {})

    # Get dataset configuration
    tokens_dir = data_config.get("tokens_dir", "data/tokens")
    dataset_version = data_config.get("dataset_version", "mimic-iii")

    logger.info(f"Loading tokenized {dataset_version} datasets from {tokens_dir}")

    # Create path to the consolidated DuckDB file
    tokens_db_path = Path(tokens_dir) / f"{dataset_version}-tokens.duckdb"
    logger.info(f"Loading all tokenized datasets from {tokens_db_path}")

    # Load train data
    logger.info("Loading tokenized training data")
    train_examples = load_tokenized_from_duckdb(
        db_path=tokens_db_path,
        dataset_version=dataset_version,
        split="train"
    )

    # Load validation data
    logger.info("Loading tokenized validation data")
    val_examples = load_tokenized_from_duckdb(
        db_path=tokens_db_path,
        dataset_version=dataset_version,
        split="validation"
    )

    # Load test data
    logger.info("Loading tokenized test data")
    test_examples = load_tokenized_from_duckdb(
        db_path=tokens_db_path,
        dataset_version=dataset_version,
        split="test"
    )

    # Create datasets dictionary
    datasets = {
        "train": train_examples,
        "validation": val_examples,
        "test": test_examples
    }

    logger.info(f"Loaded {len(datasets.get('train', []))} training examples")
    logger.info(f"Loaded {len(datasets.get('validation', []))} validation examples")
    logger.info(f"Loaded {len(datasets.get('test', []))} test examples")

    return datasets


def main() -> None:
    """Main training function."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(log_level=log_level)

    # Load configuration
    config_path = args.config
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override configuration with command line arguments
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
        logger.info(f"Overriding output directory: {args.output_dir}")

    if args.data_path:
        config["data"]["dataset_path"] = args.data_path
        logger.info(f"Overriding DuckDB database path: {args.data_path}")

    if args.dataset_version:
        config["data"]["dataset_version"] = args.dataset_version
        logger.info(f"Overriding dataset version: {args.dataset_version}")

    if args.device:
        config["training"]["device"] = args.device
        logger.info(f"Overriding device: {args.device}")

    if args.save_checkpoint_epochs:
        config["training"]["save_checkpoint_epochs"] = args.save_checkpoint_epochs
        logger.info(f"Overriding save checkpoint epochs: {args.save_checkpoint_epochs}")

    # Handle tokenized data configuration
    if args.tokens_dir:
        config["data"]["tokens_dir"] = args.tokens_dir
        logger.info(f"Overriding tokens directory: {args.tokens_dir}")

    # Set resume_from_checkpoint in config if provided
    if args.resume_from_checkpoint:
        # Check if the checkpoint exists
        checkpoint_path = Path(args.resume_from_checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {args.resume_from_checkpoint}")
            sys.exit(1)

        config["training"]["resume_from_checkpoint"] = args.resume_from_checkpoint
        logger.info(f"Will resume training from checkpoint: {args.resume_from_checkpoint}")

    # Ensure checkpoint directory exists if specified
    checkpoint_dir = config["training"].get("checkpoint_dir", "")
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Load the data
    logger.info("Loading data")
    datasets = load_data(config)

    # Create the trainer from config file
    logger.info("Creating trainer")
    trainer = create_trainer_from_config(args.config)

    # Load the model and prepare for LoRA
    trainer.load_model()
    trainer.prepare_for_lora()

    # Get train and validation datasets
    train_dataset = datasets.get("train")
    eval_dataset = datasets.get("validation")

    logger.info("Starting training")
    trainer.train(train_dataset, eval_dataset)

    # The model is automatically saved during training
    logger.info("Training completed")
    output_dir = config["training"]["output_dir"]
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
