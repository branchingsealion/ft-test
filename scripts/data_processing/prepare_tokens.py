#!/usr/bin/env python
"""
Script to prepare tokenized datasets for training.

This script loads the split datasets from the DuckDB created by prepare_mimic.py,
applies the same preprocessing logic as in the prepare_datasets function from train.py,
and saves the preprocessed data into separate DuckDB instances for train, val, and test.

Features:
- Loads train, validation, and test splits from the DuckDB database created by prepare_mimic.py
- Applies preprocessing logic to create prompt-completion pairs and tokenize them
- Saves the preprocessed data into separate DuckDB instances for train, val, and test
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import json
import duckdb
from tqdm.contrib.concurrent import thread_map

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader_duckdb import load_from_duckdb, prepare_training_examples
from src.data.processor import MedicalDataPreprocessor
from src.utils.logging_utils import setup_logger

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare tokenized datasets for training"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/training-config.yaml",
        help="Path to the training configuration file"
    )

    parser.add_argument(
        "--input-db",
        type=str,
        help="Path to the input DuckDB database created by prepare_mimic.py"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the output DuckDB databases"
    )

    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["mimic-iii", "mimic-iv"],
        help="Version of the MIMIC dataset to process"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

def load_data(config: dict, input_db: str = None, dataset_version: str = None) -> dict:
    """
    Load and prepare the training data from DuckDB.
    
    Args:
        config: Configuration dictionary
        input_db: Path to the input DuckDB database (overrides config)
        dataset_version: Version of the MIMIC dataset (overrides config)
    
    Returns:
        dictionary containing the training, validation, and test datasets
    """
    data_config = config.get("data", {})
    
    # Get dataset configuration
    dataset_path = input_db or data_config.get("dataset_path", "data/mimic-dataset.duckdb")
    dataset_version = dataset_version or data_config.get("dataset_version", "mimic-iii")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {dataset_version} dataset from DuckDB database at {dataset_path}")
    
    # Load train data
    logger.info("Loading training data from pre-existing split")
    train_data = load_from_duckdb(
        db_path=dataset_path,
        dataset_version=dataset_version,
        split="train"
    )
    train_examples = prepare_training_examples(train_data)
    
    # Load validation data
    logger.info("Loading validation data from pre-existing split")
    val_data = load_from_duckdb(
        db_path=dataset_path,
        dataset_version=dataset_version,
        split="val"
    )
    val_examples = prepare_training_examples(val_data)
    
    # Load test data
    logger.info("Loading test data from pre-existing split")
    test_data = load_from_duckdb(
        db_path=dataset_path,
        dataset_version=dataset_version,
        split="test"
    )
    test_examples = prepare_training_examples(test_data)
    
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

def prepare_datasets(datasets: dict, config: dict) -> dict:
    """
    Prepare the datasets for training.
    
    Args:
        datasets: dictionary containing the datasets
        config: Configuration dictionary
    
    Returns:
        dictionary containing the prepared datasets with input_ids, attention_mask, and labels
        for causal language modeling
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_config = config.get("model", {})
    model_name = model_config.get("model_name", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    max_length = model_config.get("max_length", 512)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing datasets with model {model_name}")
    
    # Initialize the preprocessor
    preprocessor = MedicalDataPreprocessor(model_name, max_length)
    
    # Prepare the datasets
    prepared_datasets = {}
    
    for split, examples in datasets.items():
        logger.info(f"Preparing {split} dataset")
        
        # Create prompt-completion pairs
        pairs = preprocessor.batch_create_prompt_completion_pairs(examples)
        
        # Process pairs for causal language modeling
        processed_examples = thread_map(
            preprocessor.preprocess_for_causal_lm,
            pairs,
            chunksize=512,
            desc=f"Processing {split} pairs for causal language modeling"
        )
        
        prepared_datasets[split] = processed_examples
        
        logger.info(f"Prepared {len(processed_examples)} examples for {split} dataset")
    
    return prepared_datasets

def save_to_duckdb(split: str, examples: list, conn, schema: str):
    """
    Save a single dataset split to a DuckDB table.
    
    Args:
        split: the dataset split name (train, validation, test)
        examples: list of examples for the split
        conn: DuckDB connection
        schema: schema name in the database
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Saving {split} dataset to table {schema}.{split}_tokenized_data")
    
    # Create a table for the tokenized data
    conn.execute(f"""
        CREATE OR REPLACE TABLE {schema}.{split}_tokenized_data (
            id INTEGER,
            input_ids BLOB,
            attention_mask BLOB,
            labels BLOB
        )
    """)
    
    # Insert the examples into the table
    for i, example in enumerate(examples):
        # Convert tensors to JSON strings
        input_ids_json = json.dumps(example["input_ids"].tolist())
        attention_mask_json = json.dumps(example["attention_mask"].tolist())
        labels_json = json.dumps(example["labels"].tolist())
        
        conn.execute(f"""
            INSERT INTO {schema}.{split}_tokenized_data (id, input_ids, attention_mask, labels)
            VALUES (?, ?, ?, ?)
        """, [i, input_ids_json, attention_mask_json, labels_json])
    
    logger.info(f"Saved {len(examples)} examples to table {schema}.{split}_tokenized_data")

def main():
    """
    Main function to prepare tokenized datasets for training.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(__name__, logging.getLevelName(log_level))
    
    # Load configuration
    config_path = args.config
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override configuration with command line arguments
    if args.input_db:
        logger.info(f"Overriding input database path: {args.input_db}")
    
    if args.output_dir:
        config["data"]["tokens_dir"] = args.output_dir
        logger.info(f"Overriding output directory: {args.output_dir}")
    
    if args.dataset_version:
        config["data"]["dataset_version"] = args.dataset_version
        logger.info(f"Overriding dataset version: {args.dataset_version}")
    
    # Get output directory
    output_dir = args.output_dir or config["data"].get("tokens_dir", "data/tokens")
    
    # Get dataset version
    dataset_version = args.dataset_version or config["data"].get("dataset_version", "mimic-iii")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a single DuckDB file for all splits
    db_path = output_path / f"{dataset_version}-tokens.duckdb"
    logger.info(f"Creating database at {db_path}")
    
    # Create a new DuckDB connection
    conn = duckdb.connect(str(db_path))
    
    # Create a schema for the dataset
    schema = dataset_version.lower().replace("-", "_")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    
    try:
        # Load the data
        logger.info("Loading data")
        datasets = load_data(config, args.input_db, args.dataset_version)
        
        # Process and save each split one at a time
        for split in datasets.keys():
            logger.info(f"Processing and saving {split} split")
            
            # Prepare the dataset for this split
            split_data = {split: datasets[split]}
            prepared_split = prepare_datasets(split_data, config)
            
            # Save the prepared dataset to DuckDB
            save_to_duckdb(split, prepared_split[split], conn, schema)
            
            # Clear memory
            del prepared_split
            
        logger.info(f"All datasets saved to {db_path}")
    finally:
        # Close the connection
        conn.close()
    
    logger.info("Data preparation completed")

if __name__ == "__main__":
    main()