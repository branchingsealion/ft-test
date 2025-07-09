"""
Data loading utilities for medical text and code datasets using DuckDB.

This module provides functions to load and preprocess medical text data
and associated ICD/HCC codes from a DuckDB database for model training and inference.
"""

import logging
from pathlib import Path
from typing import Any

import duckdb
import pdfplumber
import fireducks.pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Define database paths
DB_DIR = Path("data")
DB_FILE = "mimic-dataset.duckdb"
DB_PATH = DB_DIR / DB_FILE


def load_from_duckdb(
    db_path: str | Path = DB_PATH,
    dataset_version: str = "mimic-iii",
    split: str | None = None
) -> list[dict[str, Any]]:
    """
    Load data from a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')
        split: Data split to load ('train', 'val', 'test', or None for all data)

    Returns:
        List of dictionaries containing the dataset examples

    Raises:
        ValueError: If the dataset version is not supported
        FileNotFoundError: If the database file cannot be found
    """
    db_path = Path(db_path)
    logger.info(f"Loading {dataset_version} dataset from DuckDB database at {db_path}")

    if not db_path.exists():
        logger.error(f"DuckDB database not found at {db_path}")
        raise FileNotFoundError(f"DuckDB database not found at {db_path}")

    if dataset_version.lower() not in ["mimic-iii", "mimic-iv"]:
        logger.error(f"Unsupported dataset version: {dataset_version}")
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Use 'mimic-iii' or 'mimic-iv'.")

    # Create a connection to the database
    conn = duckdb.connect(str(db_path))
    
    try:
        # Determine the schema and table/view to query
        schema = dataset_version.lower().replace("-", "_")
        
        # Build the query based on the split
        if split:
            if split.lower() == "train":
                split_table = f"{schema}.train_full"
            elif split.lower() in ["val", "validation"]:
                split_table = f"{schema}.val_full"
            elif split.lower() == "test":
                split_table = f"{schema}.test_full"
            else:
                logger.error(f"Unsupported split: {split}")
                raise ValueError(f"Unsupported split: {split}. Use 'train', 'val', or 'test'.")
                
            # Use the text column directly from the view
            query = f"""
                SELECT s.* FROM {split_table} s
            """
        else:
            # Use the text column directly from the all_data view
            query = f"""
                SELECT s.* FROM {schema}.all_data s
            """
        
        # Execute the query
        logger.info(f"Executing query: {query}")
        result = conn.execute(query).fetchdf()
        
        # Convert DataFrame to list of dictionaries
        examples = []
        
        for _, row in result.iterrows():
            # Extract diagnosis and procedure codes
            diagnosis_codes = row.get('diagnosis_codes', [])
            procedure_codes = row.get('procedure_codes', [])
            
            # Determine the ICD version based on the dataset version
            icd_version = "ICD-9" if dataset_version.lower() == "mimic-iii" else "ICD-10"
            
            # Create the example dictionary
            example = {
                "text": row.get('TEXT', ''),
                "codes": [f"{icd_version}:{code}" for code in diagnosis_codes] + 
                         [f"{icd_version}:{code}" for code in procedure_codes],
                "metadata": {
                    "source": dataset_version.lower(),
                    "hadm_id": row.get('HADM_ID'),
                    "subject_id": row.get('SUBJECT_ID'),
                }
            }
            
            # Add additional metadata based on dataset version
            if dataset_version.lower() == "mimic-iii":
                example["metadata"]["category"] = row.get('CATEGORY')
            else:  # MIMIC-IV
                example["metadata"]["note_type"] = row.get('NOTE_TYPE')
            
            examples.append(example)
        
        logger.info(f"Loaded {len(examples)} examples from {dataset_version} dataset")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading data from DuckDB: {str(e)}")
        raise
    
    finally:
        # Close the connection
        conn.close()

def extract_text_from_pdf(pdf_path: str | Path) -> dict[str, Any]:
    """
    Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary containing the extracted text and metadata

    Raises:
        FileNotFoundError: If the PDF file does not exist
        ValueError: If the PDF cannot be processed
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting text from PDF: {pdf_path}")

    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        start_time = pd.Timestamp.now()
        with pdfplumber.open(str(pdf_path)) as pdf:
            content = ""
            for page in pdf.pages:
                content += page.extract_text() or ""

            extraction_time = (pd.Timestamp.now() - start_time).total_seconds()

            return {
                "content": content,
                "metadata": {
                    "filename": pdf_path.name,
                    "page_count": len(pdf.pages),
                    "extraction_time": extraction_time
                },
                "sections": []  # Can be populated by a section parser if needed
            }
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Could not process PDF: {str(e)}")


def prepare_training_examples(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Prepare examples for model training by formatting text and codes.

    Args:
        data: List of data examples with text and codes

    Returns:
        List of formatted training examples
    """
    logger.info(f"Preparing {len(data)} training examples")

    training_examples = []

    for item in data:
        # Extract text and codes
        text = item.get("text", "")
        codes = item.get("codes", [])
        
        # Skip examples with empty text or codes
        if not text or not codes:
            continue
        
        # Create a training example
        example = {
            "text": text,
            "codes": codes,
            "metadata": item.get("metadata", {})
        }
        
        training_examples.append(example)
    
    logger.info(f"Prepared {len(training_examples)} training examples")
    return training_examples
