"""
Data loading utilities for tokenized datasets from DuckDB.

This module provides functions to load tokenized datasets from DuckDB databases
created by the prepare_tokens.py script.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List

import duckdb
import torch

# Configure logging
logger = logging.getLogger(__name__)

def load_tokenized_from_duckdb(
    db_path: str | Path,
    dataset_version: str = "mimic-iii",
    split: str = "train"
) -> List[Dict[str, torch.Tensor]]:
    """
    Load tokenized data from a DuckDB database created by prepare_tokens.py.

    Args:
        db_path: Path to the DuckDB database file
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')
        split: Data split to load ('train', 'validation', or 'test')

    Returns:
        List of dictionaries containing the tokenized examples with input_ids, attention_mask, and labels

    Raises:
        ValueError: If the dataset version is not supported or if the split is invalid
        FileNotFoundError: If the database file cannot be found
    """
    db_path = Path(db_path)
    logger.info(f"Loading tokenized {split} dataset from DuckDB database at {db_path}")

    if not db_path.exists():
        logger.error(f"DuckDB database not found at {db_path}")
        raise FileNotFoundError(f"DuckDB database not found at {db_path}")

    if dataset_version.lower() not in ["mimic-iii", "mimic-iv"]:
        logger.error(f"Unsupported dataset version: {dataset_version}")
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Use 'mimic-iii' or 'mimic-iv'.")
    
    if split not in ["train", "validation", "test"]:
        logger.error(f"Unsupported split: {split}")
        raise ValueError(f"Unsupported split: {split}. Use 'train', 'validation', or 'test'.")

    # Create a connection to the database
    conn = duckdb.connect(str(db_path))
    
    try:
        # Determine the schema
        schema = dataset_version.lower().replace("-", "_")
        
        # Query the tokenized data from the appropriate view based on the split
        query = f"""
            SELECT * FROM {schema}.{split}_tokenized_view
            ORDER BY id
        """
        
        # Execute the query
        logger.info(f"Executing query: {query}")
        result = conn.execute(query).fetchdf()
        
        # Convert DataFrame to list of dictionaries
        examples = []
        
        for _, row in result.iterrows():
            # Parse JSON strings back to lists
            input_ids_list = json.loads(row['input_ids'])
            attention_mask_list = json.loads(row['attention_mask'])
            labels_list = json.loads(row['labels'])
            
            # Convert lists to PyTorch tensors
            example = {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
                "labels": torch.tensor(labels_list, dtype=torch.long)
            }
            
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} tokenized examples from {db_path}")
        
        return examples
    
    finally:
        # Close the connection
        conn.close()