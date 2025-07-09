"""
Data loading utilities for medical text and code datasets.

This module provides functions to load and preprocess medical text data
and associated ICD/HCC codes for model training and inference.
"""

import logging
from pathlib import Path
from typing import Any

import fireducks.pandas as pd
import pdfplumber

# Configure logging
logger = logging.getLogger(__name__)

def load_dataset(file_path: str | Path) -> list[dict[str, Any]]:
    """
    Load a dataset from a CSV or JSON file.

    Args:
        file_path: Path to the dataset file

    Returns:
        List of dictionaries containing the dataset examples

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file format is not supported
    """
    file_path = Path(file_path)
    logger.info(f"Loading dataset from {file_path}")

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    file_ext = file_path.suffix.lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_ext == '.json':
            return pd.read_json(file_path).to_dict('records')
        else:
            logger.error(f"Unsupported file format: {file_ext}")
            raise ValueError(f"Unsupported file format: {file_ext}")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

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
        if "text" not in item or "codes" not in item:
            logger.warning(f"Skipping example missing required fields: {item}")
            continue

        # Create a training example with the required format
        example = {
            "text": item["text"],
            "codes": item["codes"],
            "metadata": item.get("metadata", {})
        }

        training_examples.append(example)

    logger.info(f"Created {len(training_examples)} training examples")
    return training_examples

def split_dataset(data: list[dict[str, Any]], 
                 train_ratio: float = 0.8, 
                 val_ratio: float = 0.1, 
                 test_ratio: float = 0.1,
                 random_seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """
    Split a dataset into training, validation, and test sets.

    Args:
        data: List of data examples
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing the split datasets

    Raises:
        ValueError: If the ratios don't sum to 1.0
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        logger.error("Dataset split ratios must sum to 1.0")
        raise ValueError("Dataset split ratios must sum to 1.0")

    logger.info(f"Splitting dataset with ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Shuffle the data
    import random
    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split indices
    n = len(shuffled_data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split the data
    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]

    logger.info(f"Split dataset into {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test examples")

    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def load_from_duckdb(db_path: str | Path, dataset_version: str = "mimic-iii") -> list[dict[str, Any]]:
    """
    Load data from a DuckDB database.

    Args:
        db_path: Path to the DuckDB database file
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')

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

    # Create a connection string for DuckDB
    conn_str = f"duckdb:///{db_path}"
    schema = dataset_version.lower().replace("-", "_")

    try:
        # Load data from the database
        logger.info(f"Loading data from {schema} schema")
        
        # Read from the all_data view
        query = f"SELECT * FROM {schema}.all_data"
        df = pd.read_sql(query, conn_str)
        
        logger.info(f"Loaded {len(df)} rows from {schema}.all_data")
        
        # Convert DataFrame to list of dictionaries
        examples = df.to_dict('records')
        
        logger.info(f"Loaded {len(examples)} examples from {dataset_version} dataset")
        return examples
    
    except Exception as e:
        logger.error(f"Error loading data from DuckDB: {str(e)}")
        raise

def load_mimic_dataset(dataset_path: str | Path, 
                      dataset_version: str = "mimic-iv",
                      notes_table: str | None = None,
                      codes_table: str | None = None,
                      use_duckdb: bool = False) -> list[dict[str, Any]]:
    """
    Load MIMIC-III or MIMIC-IV dataset for medical code prediction.

    Args:
        dataset_path: Path to the MIMIC dataset directory or DuckDB database file
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')
        notes_table: Optional path to the notes table (relative to dataset_path)
        codes_table: Optional path to the codes table (relative to dataset_path)
        use_duckdb: Whether to load data from a DuckDB database

    Returns:
        List of dictionaries containing the dataset examples with text and codes

    Raises:
        ValueError: If the dataset version is not supported
        FileNotFoundError: If the dataset files cannot be found
    """
    # If using DuckDB, load data from the database
    if use_duckdb:
        return load_from_duckdb(dataset_path, dataset_version)
    
    # Otherwise, load data from CSV files
    dataset_path = Path(dataset_path)
    logger.info(f"Loading {dataset_version} dataset from {dataset_path}")

    if dataset_version.lower() not in ["mimic-iii", "mimic-iv"]:
        logger.error(f"Unsupported dataset version: {dataset_version}")
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Use 'mimic-iii' or 'mimic-iv'.")

    # Set default paths based on dataset version
    if dataset_version.lower() == "mimic-iii":
        notes_table = notes_table or "NOTEEVENTS.csv"
        codes_table = codes_table or "DIAGNOSES_ICD.csv"

        # MIMIC-III specific paths
        notes_path = dataset_path / notes_table
        codes_path = dataset_path / codes_table

        if not notes_path.exists() or not codes_path.exists():
            logger.error(f"MIMIC-III files not found at {dataset_path}")
            raise FileNotFoundError(f"MIMIC-III files not found. Expected {notes_path} and {codes_path}")

        # Load MIMIC-III notes
        logger.info(f"Loading MIMIC-III notes from {notes_path}")
        notes_df = pd.read_csv(notes_path)

        # Load MIMIC-III diagnosis codes
        logger.info(f"Loading MIMIC-III diagnosis codes from {codes_path}")
        codes_df = pd.read_csv(codes_path)

        # Process MIMIC-III data
        # This is a simplified example - in a real implementation,
        # you would need to join the tables and process the data
        examples = []

        # Group codes by hadm_id
        codes_by_hadm = codes_df.groupby('hadm_id')['icd9_code'].apply(list).to_dict()

        # Match notes with codes
        for _, note in notes_df.iterrows():
            hadm_id = note.get('hadm_id')
            if hadm_id and hadm_id in codes_by_hadm:
                examples.append({
                    "text": note.get('text', ''),
                    "codes": [f"ICD-9:{code}" for code in codes_by_hadm[hadm_id]],
                    "metadata": {
                        "source": "mimic-iii",
                        "hadm_id": hadm_id,
                        "subject_id": note.get('subject_id'),
                        "category": note.get('category')
                    }
                })

    else:  # MIMIC-IV
        notes_table = notes_table or "NOTE/DISCHARGE.csv"
        codes_table = codes_table or "HOSP/DIAGNOSES_ICD.csv"

        # MIMIC-IV specific paths
        notes_path = dataset_path / notes_table
        codes_path = dataset_path / codes_table

        if not notes_path.exists() or not codes_path.exists():
            logger.error(f"MIMIC-IV files not found at {dataset_path}")
            raise FileNotFoundError(f"MIMIC-IV files not found. Expected {notes_path} and {codes_path}")

        # Load MIMIC-IV notes
        logger.info(f"Loading MIMIC-IV notes from {notes_path}")
        notes_df = pd.read_csv(notes_path)

        # Load MIMIC-IV diagnosis codes
        logger.info(f"Loading MIMIC-IV diagnosis codes from {codes_path}")
        codes_df = pd.read_csv(codes_path)

        # Process MIMIC-IV data
        # This is a simplified example - in a real implementation,
        # you would need to join the tables and process the data
        examples = []

        # Group codes by hadm_id
        codes_by_hadm = codes_df.groupby('hadm_id')['icd_code'].apply(list).to_dict()

        # Match notes with codes
        for _, note in notes_df.iterrows():
            hadm_id = note.get('hadm_id')
            if hadm_id and hadm_id in codes_by_hadm:
                examples.append({
                    "text": note.get('text', ''),
                    "codes": [f"ICD-10:{code}" for code in codes_by_hadm[hadm_id]],
                    "metadata": {
                        "source": "mimic-iv",
                        "hadm_id": hadm_id,
                        "subject_id": note.get('subject_id'),
                        "note_type": note.get('note_type')
                    }
                })

    logger.info(f"Loaded {len(examples)} examples from {dataset_version}")
    return examples
