# Prompt for Generating Medical Code Prediction Project

## Project Overview

Create a Python project called "ft-test" for fine-tuning large language models to predict ICD codes from medical text using LoRA (Low-Rank Adaptation). The system processes MIMIC-III or MIMIC-IV datasets, prepares them for training using DuckDB for efficient data storage and processing, fine-tunes a language model, and provides inference capabilities through both a command-line interface and a REST API.

## Project Structure

Create the following directory structure:

```
ft-test/
├── config/
│   ├── inference-config.yaml
│   └── training-config.yaml
├── data/
│   └── tokens/
├── output/
│   └── medical_coder_model/
├── scripts/
│   ├── data_processing/
│   │   ├── prepare_mimic.py
│   │   └── prepare_tokens.py
│   ├── deployment/
│   │   ├── export_model.py
│   │   ├── predict.py
│   │   ├── start_api.py
│   │   └── test_export.py
│   └── training/
│       └── train.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── loader_duckdb.py
│   │   ├── loader_tokenized.py
│   │   └── processor.py
│   ├── deploy/
│   │   ├── __init__.py
│   │   └── api.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── export/
│   │   │   ├── __init__.py
│   │   │   └── exporter.py
│   │   ├── inference.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
├── tests/
│   └── test_api.py
├── README.md
├── justfile
├── pyproject.toml
└── poetry.lock
```

## Key Components

### 1. Data Processing

Implement data processing scripts for MIMIC-III and MIMIC-IV datasets:

- `scripts/data_processing/prepare_mimic.py`: Script to load MIMIC datasets into DuckDB and prepare them for training
  - Loads CSV files into DuckDB tables
  - Creates training, validation, and test splits (80/10/10)
  - Limits processing to a configurable number of patients for efficiency
  - Creates views for use with loader_duckdb.py

- `scripts/data_processing/prepare_tokens.py`: Script to tokenize the processed data for training
  - Loads data from DuckDB
  - Creates prompt-completion pairs
  - Tokenizes the pairs for causal language modeling
  - Saves the tokenized data back to DuckDB

- `src/data/loader_duckdb.py`: Functions to load data from DuckDB
- `src/data/loader_tokenized.py`: Functions to load tokenized data for training
- `src/data/processor.py`: Functions to preprocess medical text data for training and inference

### 2. Model Training

Implement model training using LoRA for efficient fine-tuning:

- `src/model/trainer.py`: Class for fine-tuning language models using LoRA
  - Loads the base model and tokenizer
  - Prepares the model for LoRA fine-tuning
  - Trains the model on the tokenized data
  - Evaluates the model on the validation data
  - Saves the fine-tuned model and metrics

- `scripts/training/train.py`: Script for fine-tuning models
  - Parses command-line arguments
  - Loads configuration from YAML file
  - Loads tokenized data
  - Creates a trainer and trains the model

### 3. Inference

Implement inference capabilities:

- `src/model/inference.py`: Class for making predictions using fine-tuned models
  - Loads the fine-tuned model and tokenizer
  - Provides methods for predicting with Transformers and LiteLLM
  - Parses codes from model output
  - Supports batch prediction

- `scripts/deployment/predict.py`: Script for making predictions
  - Supports both text and PDF input
  - Supports both Transformers and LiteLLM for inference
  - Saves prediction results to file or prints to stdout

### 4. Model Export

Implement model export capabilities:
- `src/model/export/`: Functions for exporting models to different formats
- Support for LM Studio (GGUF format)

### 5. API

Implement a REST API for serving predictions:
- `src/deploy/api.py`: FastAPI implementation for serving predictions
- Endpoints for health check, model information, and predictions from text and PDF

### 6. Command-line Scripts

Implement command-line scripts for data processing, training, inference, and API:

- Data Processing Scripts:
  - `scripts/data_processing/prepare_mimic.py`: Script for preparing MIMIC datasets
  - `scripts/data_processing/prepare_tokens.py`: Script for tokenizing processed data

- Training Scripts:
  - `scripts/training/train.py`: Script for fine-tuning models

- Deployment Scripts:
  - `scripts/deployment/predict.py`: Script for making predictions
  - `scripts/deployment/export_model.py`: Script for exporting models
  - `scripts/deployment/start_api.py`: Script for starting the API server
  - `scripts/deployment/test_export.py`: Script for testing exported models

### 7. Configuration

Implement configuration files:

- `config/training-config.yaml`: Configuration for model training
  - Model configuration (model name, LoRA parameters, etc.)
  - Training configuration (epochs, batch size, etc.)
  - Data configuration (dataset version, paths, etc.)
  - Logging configuration

- `config/inference-config.yaml`: Configuration for model inference and API
  - Model configuration (model path, device, etc.)
  - Inference parameters (temperature, top_p, etc.)
  - API configuration (host, port, etc.)
  - Logging configuration

## Technical Requirements

1. **Base Model**: Use "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" as the base model
2. **Fine-tuning**: Implement LoRA for efficient fine-tuning
3. **Data Storage**: Use DuckDB for efficient data storage and processing
4. **Inference**: Support both Transformers and LiteLLM for inference
5. **Input Formats**: Support both text and PDF input for inference
6. **Output Formats**: Return ICD codes with descriptions
7. **API**: Implement a FastAPI-based REST API
8. **Caching**: Implement model caching to avoid re-downloading models
9. **Hardware Acceleration**: Support CUDA, MPS, and CPU for training and inference

## Dependencies

Include the following dependencies in the `pyproject.toml` file:

```toml
[tool.poetry]
name = "ft-test"
version = "0.1.0"
description = "Fine-tuning language models for medical code prediction"
authors = ["Your Name <your.email@example.com>"]

[tool.poetry.dependencies]
python = "^3.11"
transformers = "^4.38.0"
peft = "^0.7.0"
torch = "^2.1.0"
accelerate = "^0.25.0"
duckdb = "^0.9.0"
fireducks = "^0.1.0"
pandas = "^2.1.0"
numpy = "^1.24.0"
pyyaml = "^6.0.0"
fastapi = "^0.104.0"
uvicorn = "^0.23.0"
litellm = "^0.1.0"
pypdf = "^3.15.0"
psutil = "^5.9.0"
scikit-learn = "^1.3.0"
tqdm = "^4.66.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
black = "^23.9.0"
isort = "^5.12.0"
flake8 = "^6.1.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## Documentation

Create a comprehensive README.md that includes:
1. Project overview
2. Installation instructions
3. Usage instructions for:
   - Downloading and preparing data
   - Fine-tuning models
   - Making predictions
   - Starting the API server
   - Exporting models
4. API endpoint documentation
5. Configuration options
6. Testing instructions

## Implementation Details

### Data Processing

#### MIMIC Data Preparation

1. **Loading MIMIC Data**:
   - Load MIMIC-III or MIMIC-IV CSV files into DuckDB tables
   - Support both dataset versions with appropriate table structures
   - Skip existing tables unless explicitly requested to drop them

2. **Patient Selection**:
   - Limit processing to a configurable number of patients (SUBJECT_ID)
   - Select patients with discharge notes first, then process only their data
   - This ensures complete data for a specific number of patients

3. **Data Processing**:
   - Create views for discharge notes, diagnoses, and procedures
   - Group diagnosis and procedure codes by patient and hospital admission
   - Create a fact table with notes, codes, and patient information

4. **Data Splitting**:
   - Create 80/10/10 train/validation/test splits
   - Store splits in the same DuckDB database
   - Create views to reconstruct full datasets for each split

#### Tokenization

1. **Loading Split Data**:
   - Load train, validation, and test splits from DuckDB
   - Prepare training examples with notes and codes

2. **Creating Prompt-Completion Pairs**:
   - Format examples as prompt-completion pairs for causal language modeling
   - Prompt: "Predict the medical codes for the following text: {note}"
   - Completion: Comma-separated list of ICD codes

3. **Tokenization**:
   - Tokenize the pairs using the model's tokenizer
   - Process in parallel for efficiency
   - Save tokenized data back to DuckDB

### Model Training

1. **Model Loading**:
   - Load the base model and tokenizer
   - Configure device mapping based on available hardware (CUDA, MPS, CPU)
   - Ensure the tokenizer has a padding token

2. **LoRA Configuration**:
   - Prepare the model for k-bit training
   - Configure LoRA with parameters from the configuration (r, alpha, dropout)
   - Apply LoRA to the model using the PEFT library

3. **Training Loop**:
   - Set up training arguments (epochs, batch size, learning rate, etc.)
   - Create a data collator for causal language modeling
   - Initialize a Hugging Face Trainer
   - Train the model with support for checkpointing and resuming
   - Save the final model and metrics

### Inference

1. **Model Loading**:
   - Load the base model and tokenizer
   - Load the LoRA adapter
   - Configure device mapping based on available hardware

2. **Prediction with Transformers**:
   - Format input with instruction: "Predict the medical codes for the following text:"
   - Tokenize input and generate output using the model
   - Extract codes from the output
   - Return a dictionary with codes, model version, and processing time

3. **Prediction with LiteLLM**:
   - Format input with the same instruction
   - Use LiteLLM to make a completion request to the local model
   - Extract codes from the output
   - Return a dictionary with codes, model version, and processing time

4. **Code Parsing**:
   - Split output by commas and clean up
   - Filter out empty strings and non-code text
   - Return a list of valid codes

5. **Batch Prediction**:
   - Process texts in batches for efficiency
   - Return a list of prediction results

### API

1. **FastAPI Implementation**:
   - Create a FastAPI application
   - Load the model at startup
   - Implement endpoints for health check, model information, and predictions
   - Support both text and PDF input
   - Return predictions in a structured format

2. **API Server**:
   - Configure host, port, and CORS settings
   - Implement rate limiting and timeout settings
   - Start the server with uvicorn

## Command-Line Usage

### Data Processing

```bash
# Prepare MIMIC-III dataset
python scripts/data_processing/prepare_mimic.py \
  --mimic-dir /path/to/mimic-iii \
  --dataset-version mimic-iii \
  --output-dir data \
  --num-samples 100

# Prepare MIMIC-IV dataset
python scripts/data_processing/prepare_mimic.py \
  --mimic-dir /path/to/mimic-iv \
  --dataset-version mimic-iv \
  --output-dir data \
  --num-samples 100

# Prepare tokenized datasets
python scripts/data_processing/prepare_tokens.py \
  --config config/training-config.yaml \
  --input-db data/mimic-dataset.duckdb \
  --output-dir data/tokens
```

### Training

```bash
# Train the model
python scripts/training/train.py \
  --config config/training-config.yaml \
  --output-dir output/medical_coder_model \
  --device cuda
```

### Inference

```bash
# Predict from text
python scripts/deployment/predict.py \
  --config config/inference-config.yaml \
  --text "Patient has a history of hypertension and type 2 diabetes." \
  --output predictions.json

# Predict from PDF
python scripts/deployment/predict.py \
  --config config/inference-config.yaml \
  --pdf path/to/document.pdf \
  --output predictions.json

# Use LiteLLM for inference
python scripts/deployment/predict.py \
  --config config/inference-config.yaml \
  --text "Patient has a history of hypertension and type 2 diabetes." \
  --use-litellm
```

### API

```bash
# Start the API server
python scripts/deployment/start_api.py \
  --config config/inference-config.yaml \
  --host 0.0.0.0 \
  --port 8000
```

## Testing

Include tests for:

1. Data processing functions
   - Test loading MIMIC data into DuckDB
   - Test creating training examples
   - Test tokenization

2. Model training functions
   - Test model loading
   - Test LoRA configuration
   - Test training loop

3. Inference functions
   - Test model loading
   - Test prediction with Transformers
   - Test prediction with LiteLLM
   - Test code parsing

4. API endpoints
   - Test health check endpoint
   - Test model information endpoint
   - Test prediction endpoints

## Additional Features

1. **Model Caching**: Implement caching to avoid re-downloading models
2. **Hardware Acceleration**: Support CUDA, MPS, and CPU for training and inference
3. **Batch Processing**: Support batch prediction for processing multiple texts efficiently
4. **Memory Tracking**: Log memory usage during training and inference
5. **Checkpoint Management**: Save and load checkpoints during training
6. **Export Formats**: Support exporting models to different formats for deployment
7. **PDF Processing**: Extract text from PDF files for inference
8. **Configurable Parameters**: Support command-line arguments and configuration files
9. **Logging**: Comprehensive logging for training, inference, and API
10. **Error Handling**: Robust error handling and fallback mechanisms
