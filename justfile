# justfile for running scripts with common command lines

# Default configuration files
default_train_config := "config/training-config.yaml"
default_inference_config := "config/inference-config.yaml"

# Train a medical code prediction model
train *ARGS:
    python scripts/training/train.py --config {{default_train_config}} {{ARGS}}

# Train with debug mode
train-debug *ARGS:
    python scripts/training/train.py --config {{default_train_config}} --debug {{ARGS}}

# Train using DuckDB for data loading
train-duckdb *ARGS:
    python scripts/training/train.py --config {{default_train_config}} --use_duckdb {{ARGS}}

# Train with DuckDB in debug mode
train-duckdb-debug *ARGS:
    python scripts/training/train.py --config {{default_train_config}} --use_duckdb --debug {{ARGS}}

# Predict medical codes from text or PDF
predict *ARGS:
    python scripts/deployment/predict.py --config {{default_inference_config}} {{ARGS}}

# Predict with debug mode
predict-debug *ARGS:
    python scripts/deployment/predict.py --config {{default_inference_config}} --debug {{ARGS}}

# Export a fine-tuned model to GGUF format
export MODEL_PATH *ARGS:
    python scripts/deployment/export_model.py --model-path {{MODEL_PATH}} {{ARGS}}

# Test model exporting functionality
test-export MODEL_PATH *ARGS:
    python scripts/deployment/test_export.py --model-path {{MODEL_PATH}} {{ARGS}}

# Start the FastAPI server for medical code prediction
api *ARGS:
    python scripts/deployment/start_api.py --config {{default_inference_config}} {{ARGS}}

# Start the API server in debug mode
api-debug *ARGS:
    python scripts/deployment/start_api.py --config {{default_inference_config}} --debug {{ARGS}}

# Prepare MIMIC dataset for training
prepare-dataset mimic_dir dataset_version="mimic-iii":
    @echo "Preparing {{dataset_version}} dataset from {{mimic_dir}}..."
    @mkdir -p data
    @python scripts/data_processing/prepare_mimic.py --mimic-dir {{mimic_dir}} --dataset-version {{dataset_version}} --output-dir data
    @echo "Dataset prepared successfully! Files created: data/train.csv, data/val.csv, data/test.csv"

# Prepare MIMIC-III dataset for training (alias for backward compatibility)
prepare-mimic mimic_dir:
    @just prepare-dataset {{mimic_dir}} mimic-iii

# Prepare MIMIC-IV dataset for training
prepare-mimic-iv mimic_dir:
    @just prepare-dataset {{mimic_dir}} mimic-iv

# Load MIMIC dataset into DuckDB
load-to-duckdb mimic_dir dataset_version="mimic-iii":
    @echo "Loading {{dataset_version}} dataset from {{mimic_dir}} into DuckDB..."
    @mkdir -p data
    @python scripts/data_processing/load_mimic_to_duckdb.py --mimic_dir {{mimic_dir}} --version {{dataset_version}}
    @echo "Dataset loaded successfully into data/mimic-dataset.duckdb"

# Load MIMIC-III dataset into DuckDB
load-mimic-to-duckdb mimic_dir:
    @just load-to-duckdb {{mimic_dir}} mimic-iii

# Load MIMIC-IV dataset into DuckDB
load-mimic-iv-to-duckdb mimic_dir:
    @just load-to-duckdb {{mimic_dir}} mimic-iv

# Examples:
# just train
# just train --data path/to/data.csv --epochs 5 --batch-size 8
# just train-duckdb --dataset_version mimic-iii
# just load-mimic-to-duckdb path/to/mimic-iii/directory
# just load-mimic-iv-to-duckdb path/to/mimic-iv/directory
# just predict --text "Patient has hypertension and diabetes"
# just predict --pdf path/to/medical_record.pdf --output results.json
# just export models/my-fine-tuned-model --output-path exports
# just test-export models/my-fine-tuned-model
# just api
# just api --host 127.0.0.1 --port 8080
