#!/usr/bin/env python
"""
Script to prepare MIMIC-III or MIMIC-IV dataset for training.

This script loads MIMIC-III or MIMIC-IV dataset tables into a DuckDB database and prepares them for training
by processing the data and creating training, validation, and test splits (80/10/10) directly in the database.

Features:
- Loads all data from CSVs into tables in a DuckDB database
- Skips tables if they already exist, with an option to drop them (--drop-tables)
- Limits processing to a specified number of patients (SUBJECT_ID) for efficiency
- Creates 80/10/10 train/validation/test splits stored in the same database
- Creates train_full, test_full, and val_full views for use with loader_duckdb.py
"""

import argparse
import logging
import sys
from logging import getLevelName
from pathlib import Path

import duckdb
import fireducks.pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logging_utils import setup_logger

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Prepare MIMIC-III or MIMIC-IV dataset for training"
    )

    parser.add_argument(
        "--mimic-dir",
        type=str,
        required=True,
        help="Path to the MIMIC dataset directory containing CSV files"
    )

    parser.add_argument(
        "--dataset-version",
        type=str,
        choices=["mimic-iii", "mimic-iv"],
        default="mimic-iii",
        help="Version of the MIMIC dataset to process (default: mimic-iii)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save processed dataset files (defaults to 'data')"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of patients (SUBJECT_ID) to process from the dataset (default: 100, minimum: 10)"
    )

    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Proportion of data to use for training (default: 0.8)"
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data to use for validation (default: 0.1)"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Proportion of data to use for testing (default: 0.1)"
    )

    parser.add_argument(
        "--random-seed",
        type=float,
        default=0.42,
        help="Random seed for reproducibility (Range -1.0 to 1.0 default: 0.42)"
    )

    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Drop existing tables in the DuckDB database before loading data (default: skip existing tables)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Validate num_samples
    if args.num_samples < 10:
        raise ValueError("--num-samples must be at least 10")

    return args

def load_mimic_tables_to_duckdb(
    mimic_dir: str | Path, 
    output_dir: str | Path, 
    dataset_version: str = "mimic-iii", 
    drop_tables: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Load required MIMIC tables directly into a DuckDB database.

    Args:
        mimic_dir: Path to the MIMIC dataset directory
        output_dir: Directory to save the DuckDB database
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')
        drop_tables: Whether to drop existing tables before loading data

    Returns:
        DuckDB connection to the database
    """
    logger = logging.getLogger(__name__)
    mimic_dir = Path(mimic_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define database path
    db_path = output_dir / "mimic-dataset.duckdb"
    
    # Create a connection to the database
    conn = duckdb.connect(str(db_path))
    
    # Create schema if it doesn't exist
    schema = dataset_version.lower().replace("-", "_")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    
    # Define table names
    tables_info = {}
    
    if dataset_version.lower() == "mimic-iii":
        logger.info(f"Loading MIMIC-III dataset from {mimic_dir} into DuckDB")

        # Check if the directory exists
        if not mimic_dir.exists():
            raise FileNotFoundError(f"MIMIC-III directory not found: {mimic_dir}")

        try:
            # Define table paths and names for MIMIC-III
            tables_info = {
                "NOTEEVENTS": {
                    "path": mimic_dir / "NOTEEVENTS.csv",
                    "table_name": f"{schema}.NOTEEVENTS",
                    "pk_column": "NOTE_PK"
                },
                "DIAGNOSES_ICD": {
                    "path": mimic_dir / "DIAGNOSES_ICD.csv",
                    "table_name": f"{schema}.DIAGNOSES_ICD",
                    "pk_column": "DIAGNOSIS_PK"
                },
                "PROCEDURES_ICD": {
                    "path": mimic_dir / "PROCEDURES_ICD.csv",
                    "table_name": f"{schema}.PROCEDURES_ICD",
                    "pk_column": "PROCEDURE_PK"
                },
                "PATIENTS": {
                    "path": mimic_dir / "PATIENTS.csv",
                    "table_name": f"{schema}.PATIENTS",
                    "pk_column": "SUBJECT_ID"  # SUBJECT_ID is already a unique identifier
                }
            }
            
            # Load each table into DuckDB
            for table_key, table_info in tables_info.items():
                csv_path = table_info["path"]
                table_name = table_info["table_name"]
                
                if not csv_path.exists():
                    logger.error(f"{csv_path.name} not found at {csv_path}")
                    raise FileNotFoundError(f"{csv_path.name} not found at {csv_path}")
                
                # Check if table exists
                table_exists = conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table_key}'
                """).fetchone()[0] > 0
                
                if table_exists:
                    if drop_tables:
                        logger.info(f"Dropping existing table {table_name}")
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    else:
                        logger.info(f"Table {table_name} already exists, skipping")
                        continue
                
                logger.info(f"Loading {table_key} from {csv_path} into {table_name}")
                
                # Load CSV into DuckDB table
                conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{csv_path}', header=true, all_varchar=false)
                """)
                
                # Convert column names to uppercase
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchdf()["name"].tolist()
                for column in columns:
                    upper_column = column.upper()
                    if column != upper_column:
                        conn.execute(f"""
                            ALTER TABLE {table_name} 
                            RENAME COLUMN "{column}" TO "{upper_column}"
                        """)
                
                # Get row count
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info(f"Loaded {table_key} with {row_count} rows")

        except Exception as e:
            logger.error(f"Error loading MIMIC-III dataset: {str(e)}")
            raise

    elif dataset_version.lower() == "mimic-iv":
        logger.info(f"Loading MIMIC-IV dataset from {mimic_dir} into DuckDB")

        # Check if the directory exists
        if not mimic_dir.exists():
            raise FileNotFoundError(f"MIMIC-IV directory not found: {mimic_dir}")

        try:
            # Define table paths and names for MIMIC-IV
            tables_info = {
                "NOTEEVENTS": {
                    "path": mimic_dir / "note" / "discharge.csv",
                    "table_name": f"{schema}.NOTEEVENTS",
                    "pk_column": "NOTE_ID"
                },
                "DIAGNOSES_ICD": {
                    "path": mimic_dir / "hosp" / "diagnoses_icd.csv",
                    "table_name": f"{schema}.DIAGNOSES_ICD",
                    "pk_column": "DIAGNOSIS_ID"
                },
                "PROCEDURES_ICD": {
                    "path": mimic_dir / "hosp" / "procedures_icd.csv",
                    "table_name": f"{schema}.PROCEDURES_ICD",
                    "pk_column": "PROCEDURE_ID"
                },
                "PATIENTS": {
                    "path": mimic_dir / "core" / "patients.csv",
                    "table_name": f"{schema}.PATIENTS",
                    "pk_column": "SUBJECT_ID"  # SUBJECT_ID is already a unique identifier
                }
            }
            
            # Load each table into DuckDB
            for table_key, table_info in tables_info.items():
                csv_path = table_info["path"]
                table_name = table_info["table_name"]
                
                if not csv_path.exists():
                    logger.error(f"{csv_path.name} not found at {csv_path}")
                    raise FileNotFoundError(f"{csv_path.name} not found at {csv_path}")
                
                # Check if table exists
                table_exists = conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table_key}'
                """).fetchone()[0] > 0
                
                if table_exists:
                    if drop_tables:
                        logger.info(f"Dropping existing table {table_name}")
                        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    else:
                        logger.info(f"Table {table_name} already exists, skipping")
                        continue
                
                logger.info(f"Loading {table_key} from {csv_path} into {table_name}")
                
                # Load CSV into DuckDB table
                conn.execute(f"""
                    CREATE TABLE {table_name} AS 
                    SELECT * FROM read_csv_auto('{csv_path}', header=true, all_varchar=false)
                """)
                
                # Convert column names to uppercase
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchdf()["name"].tolist()
                for column in columns:
                    upper_column = column.upper()
                    if column != upper_column:
                        conn.execute(f"""
                            ALTER TABLE {table_name} 
                            RENAME COLUMN "{column}" TO "{upper_column}"
                        """)
                
                # Get row count
                row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                logger.info(f"Loaded {table_key} with {row_count} rows")

        except Exception as e:
            logger.error(f"Error loading MIMIC-IV dataset: {str(e)}")
            raise

    else:
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Use 'mimic-iii' or 'mimic-iv'.")

    return conn

def process_mimic_data_in_duckdb(conn: duckdb.DuckDBPyConnection, dataset_version: str = "mimic-iii", drop_existing: bool = False, num_samples: int = 100) -> str:
    """
    Process MIMIC tables in DuckDB to create a dataset for training.

    Args:
        conn: DuckDB connection
        dataset_version: Version of the MIMIC dataset ('mimic-iii' or 'mimic-iv')
        drop_existing: Whether to drop the existing processed_data table if it exists
        num_samples: Number of patients (SUBJECT_ID) to process from the dataset (default: 100)

    Returns:
        Name of the processed dataset table
    """
    logger = logging.getLogger(__name__)
    schema = dataset_version.lower().replace("-", "_")
    processed_table = f"{schema}.processed_data"

    # Log the number of patients to be processed
    logger.info(f"Processing data for up to {num_samples} patients from the dataset")

    # Check if processed table already exists
    if drop_existing:
        conn.execute(f"DROP TABLE IF EXISTS {processed_table}")

    if dataset_version.lower() == "mimic-iii":
        logger.info("Processing MIMIC-III clinical notes in DuckDB")
        
        # Create a view for discharge summaries
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.discharge_notes AS
            SELECT * FROM {schema}.NOTEEVENTS
            WHERE CATEGORY = 'Discharge summary'
        """)
        
        # Count discharge summaries
        discharge_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.discharge_notes").fetchone()[0]
        logger.info(f"Found {discharge_count} discharge summaries")
        
        # First, select a limited number of patients who have discharge notes
        # This is a key optimization: we select patients first, then only process data for those patients
        # This approach ensures we process complete data for a specific number of patients
        # rather than arbitrarily limiting the total number of records
        logger.info(f"Selecting up to {num_samples} patients with discharge notes")
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.limited_patients AS
            SELECT DISTINCT SUBJECT_ID
            FROM {schema}.discharge_notes
            ORDER BY SUBJECT_ID
            LIMIT {num_samples}
        """)

        # Count the number of selected patients
        patient_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.limited_patients").fetchone()[0]
        logger.info(f"Selected {patient_count} patients for processing")

        logger.info("Processing MIMIC-III diagnosis codes in DuckDB")
        # Create a view with grouped diagnosis codes for the selected patients only
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.diagnoses_grouped AS
            SELECT 
                d.SUBJECT_ID, 
                d.HADM_ID, 
                list(d.ICD9_CODE) AS diagnosis_codes
            FROM {schema}.DIAGNOSES_ICD d
            JOIN {schema}.limited_patients p ON d.SUBJECT_ID = p.SUBJECT_ID
            GROUP BY d.SUBJECT_ID, d.HADM_ID
        """)
        
        logger.info("Processing MIMIC-III procedure codes in DuckDB")
        # Create a view with grouped procedure codes for the selected patients only
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.procedures_grouped AS
            SELECT 
                p.SUBJECT_ID, 
                p.HADM_ID, 
                list(p.ICD9_CODE) AS procedure_codes
            FROM {schema}.PROCEDURES_ICD p
            JOIN {schema}.limited_patients lp ON p.SUBJECT_ID = lp.SUBJECT_ID
            GROUP BY p.SUBJECT_ID, p.HADM_ID
        """)
        
        logger.info("Creating fact table for MIMIC-III data in DuckDB")
        
        # First, ensure primary keys exist on the source tables if they don't already
        # Check if NOTE_PK exists in NOTEEVENTS
        note_pk_exists = conn.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' 
            AND table_name = 'NOTEEVENTS' 
            AND column_name = 'NOTE_PK'
        """).fetchone()[0] > 0
        
        if not note_pk_exists:
            logger.info("Adding NOTE_PK to NOTEEVENTS table")
            conn.execute(f"""
                ALTER TABLE {schema}.NOTEEVENTS 
                ADD COLUMN NOTE_PK INTEGER;
                
                UPDATE {schema}.NOTEEVENTS 
                SET NOTE_PK = ROW_NUMBER() OVER ();
                
                ALTER TABLE {schema}.NOTEEVENTS
                ADD CONSTRAINT pk_noteevents PRIMARY KEY (NOTE_PK);
            """)
        
        # Create the processed data fact table that references the source tables
        # Only include data for the selected patients
        conn.execute(f"""
            CREATE TABLE {processed_table} AS
            SELECT 
                ROW_NUMBER() OVER () AS row_id,  -- Add unique row_id as primary key
                n.NOTE_PK,
                n.SUBJECT_ID,
                n.HADM_ID,
                n.CATEGORY,
                d.diagnosis_codes,
                COALESCE(p.procedure_codes, []) AS procedure_codes,
                list_cat(d.diagnosis_codes, COALESCE(p.procedure_codes, [])) AS all_codes,
                array_to_string(d.diagnosis_codes, ',') AS diagnosis_codes_str,
                array_to_string(COALESCE(p.procedure_codes, []), ',') AS procedure_codes_str,
                array_to_string(list_cat(d.diagnosis_codes, COALESCE(p.procedure_codes, [])), ',') AS all_codes_str,
                pat.GENDER AS gender
            FROM {schema}.discharge_notes n
            JOIN {schema}.limited_patients lp ON n.SUBJECT_ID = lp.SUBJECT_ID
            INNER JOIN {schema}.diagnoses_grouped d
                ON n.SUBJECT_ID = d.SUBJECT_ID AND n.HADM_ID = d.HADM_ID
            LEFT JOIN {schema}.procedures_grouped p
                ON n.SUBJECT_ID = p.SUBJECT_ID AND n.HADM_ID = p.HADM_ID
            LEFT JOIN {schema}.PATIENTS pat
                ON n.SUBJECT_ID = pat.SUBJECT_ID
        """)
        
        # Add primary key constraint to the fact table
        conn.execute(f"""
            ALTER TABLE {processed_table}
            ADD CONSTRAINT pk_processed_data PRIMARY KEY (row_id);
        """)
        
    elif dataset_version.lower() == "mimic-iv":
        logger.info("Processing MIMIC-IV clinical notes in DuckDB")
        
        # For MIMIC-IV, we need to adapt the queries to match the different table structure
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.discharge_notes AS
            SELECT * FROM {schema}.NOTEEVENTS
            WHERE NOTE_TYPE = 'DS'
        """)
        
        # Count discharge notes
        discharge_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.discharge_notes").fetchone()[0]
        logger.info(f"Found {discharge_count} discharge notes")
        
        # First, select a limited number of patients who have discharge notes
        # This is a key optimization: we select patients first, then only process data for those patients
        # This approach ensures we process complete data for a specific number of patients
        # rather than arbitrarily limiting the total number of records
        logger.info(f"Selecting up to {num_samples} patients with discharge notes")
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.limited_patients AS
            SELECT DISTINCT SUBJECT_ID
            FROM {schema}.discharge_notes
            ORDER BY SUBJECT_ID
            LIMIT {num_samples}
        """)
        
        # Count the number of selected patients
        patient_count = conn.execute(f"SELECT COUNT(*) FROM {schema}.limited_patients").fetchone()[0]
        logger.info(f"Selected {patient_count} patients for processing")
        
        logger.info("Processing MIMIC-IV diagnosis codes in DuckDB")
        # Create a view with grouped diagnosis codes for the selected patients only
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.diagnoses_grouped AS
            SELECT 
                d.SUBJECT_ID, 
                d.HADM_ID, 
                list(d.ICD_CODE) AS diagnosis_codes
            FROM {schema}.DIAGNOSES_ICD d
            JOIN {schema}.limited_patients p ON d.SUBJECT_ID = p.SUBJECT_ID
            GROUP BY d.SUBJECT_ID, d.HADM_ID
        """)
        
        logger.info("Processing MIMIC-IV procedure codes in DuckDB")
        # Create a view with grouped procedure codes for the selected patients only
        conn.execute(f"""
            CREATE OR REPLACE VIEW {schema}.procedures_grouped AS
            SELECT 
                p.SUBJECT_ID, 
                p.HADM_ID, 
                list(p.ICD_CODE) AS procedure_codes
            FROM {schema}.PROCEDURES_ICD p
            JOIN {schema}.limited_patients lp ON p.SUBJECT_ID = lp.SUBJECT_ID
            GROUP BY p.SUBJECT_ID, p.HADM_ID
        """)
        
        logger.info("Creating fact table for MIMIC-IV data in DuckDB")
        
        # First, ensure primary keys exist on the source tables if they don't already
        # Check if NOTE_ID exists in NOTEEVENTS
        note_id_exists = conn.execute(f"""
            SELECT COUNT(*) 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' 
            AND table_name = 'NOTEEVENTS' 
            AND column_name = 'NOTE_ID'
        """).fetchone()[0] > 0
        
        if not note_id_exists:
            logger.info("Adding NOTE_ID to NOTEEVENTS table")
            conn.execute(f"""
                ALTER TABLE {schema}.NOTEEVENTS 
                ADD COLUMN NOTE_ID INTEGER;
                
                UPDATE {schema}.NOTEEVENTS 
                SET NOTE_ID = ROW_NUMBER() OVER ();
                
                ALTER TABLE {schema}.NOTEEVENTS
                ADD CONSTRAINT pk_noteevents PRIMARY KEY (NOTE_ID);
            """)
        
        # Create the processed data fact table that references the source tables
        # Only include data for the selected patients
        conn.execute(f"""
            CREATE TABLE {processed_table} AS
            SELECT 
                ROW_NUMBER() OVER () AS row_id,  -- Add unique row_id as primary key
                n.NOTE_ID,
                n.SUBJECT_ID,
                n.HADM_ID,
                n.NOTE_TYPE,
                n.TEXT,
                d.diagnosis_codes,
                COALESCE(p.procedure_codes, []) AS procedure_codes,
                list_cat(d.diagnosis_codes, COALESCE(p.procedure_codes, [])) AS all_codes,
                array_to_string(d.diagnosis_codes, ',') AS diagnosis_codes_str,
                array_to_string(COALESCE(p.procedure_codes, []), ',') AS procedure_codes_str,
                array_to_string(list_cat(d.diagnosis_codes, COALESCE(p.procedure_codes, [])), ',') AS all_codes_str,
                pat.GENDER AS gender
            FROM {schema}.discharge_notes n
            JOIN {schema}.limited_patients lp ON n.SUBJECT_ID = lp.SUBJECT_ID
            INNER JOIN {schema}.diagnoses_grouped d
                ON n.SUBJECT_ID = d.SUBJECT_ID AND n.HADM_ID = d.HADM_ID
            LEFT JOIN {schema}.procedures_grouped p
                ON n.SUBJECT_ID = p.SUBJECT_ID AND n.HADM_ID = p.HADM_ID
            LEFT JOIN {schema}.PATIENTS pat
                ON n.SUBJECT_ID = pat.SUBJECT_ID
        """)
        
        # Add primary key constraint to the fact table
        conn.execute(f"""
            ALTER TABLE {processed_table}
            ADD CONSTRAINT pk_processed_data PRIMARY KEY (row_id);
        """)
        
    else:
        raise ValueError(f"Unsupported dataset version: {dataset_version}. Use 'mimic-iii' or 'mimic-iv'.")

    # Count the number of records in the processed data table
    record_count = conn.execute(f"SELECT COUNT(*) FROM {processed_table}").fetchone()[0]
    logger.info(f"Final dataset created with {record_count} records")
    
    return processed_table
def split_dataset_in_duckdb(
    conn: duckdb.DuckDBPyConnection,
    processed_table: str,
    train_size: float, 
    val_size: float, 
    test_size: float, 
    random_seed: float,
    drop_existing: bool = False
) -> tuple[str, str, str]:
    """
    Split the dataset into training, validation, and test sets directly in DuckDB.

    Args:
        conn: DuckDB connection
        processed_table: Name of the processed data table
        train_size: Proportion of data to use for training
        val_size: Proportion of data to use for validation
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
        drop_existing: Whether to drop existing split tables

    Returns:
        Tuple of (train_table, val_table, test_table) names
    """
    logger = logging.getLogger(__name__)
    
    # Extract schema from processed_table
    schema = processed_table.split('.')[0]
    
    # Define split table names
    train_table = f"{schema}.train_data"
    val_table = f"{schema}.val_data"
    test_table = f"{schema}.test_data"
    splits_table = f"{schema}.data_splits"
    
    # Check if split tables already exist
    train_exists = conn.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' 
        AND table_name = 'TRAIN_DATA'
    """).fetchone()[0] > 0
    
    val_exists = conn.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' 
        AND table_name = 'VAL_DATA'
    """).fetchone()[0] > 0
    
    test_exists = conn.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' 
        AND table_name = 'TEST_DATA'
    """).fetchone()[0] > 0
    
    splits_exists = conn.execute(f"""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' 
        AND table_name = 'DATA_SPLITS'
    """).fetchone()[0] > 0
    
    # If all split tables exist and we're not dropping them, return early
    if train_exists and val_exists and test_exists and splits_exists and not drop_existing:
        logger.info("Split tables already exist, skipping split creation")
        return train_table, val_table, test_table
    
    # Drop existing tables if requested
    if drop_existing:
        if train_exists:
            logger.info(f"Dropping existing table {train_table}")
            conn.execute(f"DROP TABLE IF EXISTS {train_table}")
        
        if val_exists:
            logger.info(f"Dropping existing table {val_table}")
            conn.execute(f"DROP TABLE IF EXISTS {val_table}")
        
        if test_exists:
            logger.info(f"Dropping existing table {test_table}")
            conn.execute(f"DROP TABLE IF EXISTS {test_table}")
        
        if splits_exists:
            logger.info(f"Dropping existing table {splits_table}")
            conn.execute(f"DROP TABLE IF EXISTS {splits_table}")
    
    # Normalize split proportions to ensure they sum to 1
    total = train_size + val_size + test_size
    train_size = train_size / total
    val_size = val_size / total
    test_size = test_size / total
    
    logger.info(f"Splitting dataset with proportions: train={train_size:.2f}, val={val_size:.2f}, test={test_size:.2f}")
    
    # Create a table with only row_id and random values for splitting
    # This approach avoids duplicating the actual data
    conn.execute(f"""
        SELECT setseed({random_seed});
        CREATE OR REPLACE TABLE {splits_table} AS
        SELECT 
            row_id,  -- Use the existing row_id from processed_table
            random() as rand_val
        FROM {processed_table}
        ORDER BY rand_val
    """)
    
    # Get total number of rows
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {splits_table}").fetchone()[0]
    train_rows = int(total_rows * train_size)
    val_rows = int(total_rows * val_size)
    
    # Create train, validation, and test tables with only row_id
    logger.info(f"Creating training set with {train_rows} rows")
    conn.execute(f"""
        CREATE TABLE {train_table} AS
        SELECT row_id
        FROM {splits_table}
        WHERE row_id <= {train_rows}
    """)
    
    logger.info(f"Creating validation set with {val_rows} rows")
    conn.execute(f"""
        CREATE TABLE {val_table} AS
        SELECT row_id
        FROM {splits_table}
        WHERE row_id > {train_rows} AND row_id <= {train_rows + val_rows}
    """)
    
    logger.info(f"Creating test set with remaining rows")
    conn.execute(f"""
        CREATE TABLE {test_table} AS
        SELECT row_id
        FROM {splits_table}
        WHERE row_id > {train_rows + val_rows}
    """)
    
    # Create views to reconstruct full datasets
    logger.info("Creating views to reconstruct full datasets")
    conn.execute(f"""
        CREATE OR REPLACE VIEW {schema}.train_full AS
        SELECT p.*
        FROM {processed_table} p
        JOIN {train_table} t ON p.row_id = t.row_id
    """)
    
    conn.execute(f"""
        CREATE OR REPLACE VIEW {schema}.val_full AS
        SELECT p.*
        FROM {processed_table} p
        JOIN {val_table} v ON p.row_id = v.row_id
    """)
    
    conn.execute(f"""
        CREATE OR REPLACE VIEW {schema}.test_full AS
        SELECT p.*
        FROM {processed_table} p
        JOIN {test_table} tst ON p.row_id = tst.row_id
    """)
    
    # Create a combined view for all data with split information
    logger.info("Creating combined view with all data and split information")
    conn.execute(f"""
        CREATE OR REPLACE VIEW {schema}.all_data AS
        SELECT p.*, 'train' as split
        FROM {processed_table} p
        JOIN {train_table} t ON p.row_id = t.row_id
        UNION ALL
        SELECT p.*, 'val' as split
        FROM {processed_table} p
        JOIN {val_table} v ON p.row_id = v.row_id
        UNION ALL
        SELECT p.*, 'test' as split
        FROM {processed_table} p
        JOIN {test_table} tst ON p.row_id = tst.row_id
    """)
    
    
    # Count rows in each split
    train_count = conn.execute(f"SELECT COUNT(*) FROM {train_table}").fetchone()[0]
    val_count = conn.execute(f"SELECT COUNT(*) FROM {val_table}").fetchone()[0]
    test_count = conn.execute(f"SELECT COUNT(*) FROM {test_table}").fetchone()[0]
    
    logger.info(f"Dataset split complete: train={train_count} rows, val={val_count} rows, test={test_count} rows")
    
    return train_table, val_table, test_table


def main() -> None:
    """
    Main entry point for the script.

    Returns:
        None
    """
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level=getLevelName(log_level))
    logger = logging.getLogger(__name__)

    logger.info(f"Starting {args.dataset_version} dataset preparation from: {args.mimic_dir}")

    try:
        # Load MIMIC tables directly into DuckDB
        conn = load_mimic_tables_to_duckdb(
            args.mimic_dir, 
            args.output_dir, 
            args.dataset_version, 
            args.drop_tables
        )
        
        # Process the data in DuckDB
        processed_table = process_mimic_data_in_duckdb(conn, args.dataset_version, args.drop_tables, args.num_samples)
        
        # Split the dataset in DuckDB
        train_table, val_table, test_table = split_dataset_in_duckdb(
            conn,
            processed_table,
            args.train_size,
            args.val_size,
            args.test_size,
            args.random_seed,
            args.drop_tables
        )
        
        # Get schema name for logging
        schema = args.dataset_version.lower().replace("-", "_")
        
        # Get row counts for summary
        train_count = conn.execute(f"SELECT COUNT(*) FROM {train_table}").fetchone()[0]
        val_count = conn.execute(f"SELECT COUNT(*) FROM {val_table}").fetchone()[0]
        test_count = conn.execute(f"SELECT COUNT(*) FROM {test_table}").fetchone()[0]
        
        # Close the database connection
        db_path = Path(args.output_dir) / "mimic-dataset.duckdb"
        conn.close()

        # Print summary
        logger.info(f"{args.dataset_version} dataset preparation completed successfully!")
        logger.info(f"Training set: {train_count} records saved to {train_table}")
        logger.info(f"Validation set: {val_count} records saved to {val_table}")
        logger.info(f"Test set: {test_count} records saved to {test_table}")
        logger.info(f"All data saved to DuckDB database at {db_path}")
        logger.info(f"Use the following to load data from DuckDB:")
        logger.info(f"  - Schema: {schema}")
        logger.info(f"  - Tables: {train_table}, {val_table}, {test_table}")
        logger.info(f"  - Views: {schema}.train_full, {schema}.val_full, {schema}.test_full, {schema}.all_data")

    except Exception as e:
        logger.exception(f"Error preparing dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()