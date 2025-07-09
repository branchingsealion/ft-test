# ft_test: Medical Code Prediction

A Python project for fine-tuning a DeepSeek models for medical code prediction using LoRA.

## Project Overview

This project implements a system for predicting ICD-10 and HCC codes from medical text using a fine-tuned DeepSeek models. 
The system includes:
- Data processing utilities for loading and preprocessing medical text data
- Model training using LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Inference using both Transformers and LiteLLM
- REST API for serving predictions
- Command-line scripts for training and inference

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ft_test.git
   cd ft_test
   ```

2. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

3. Install DuckDB dependencies (optional, but recommended for improved performance):
   ```bash
   pip install duckdb fireducks
   ```

## Usage

### Downloading the Data

To use this project, you'll need access to the MIMIC-III or MIMIC-IV dataset:

1. **Prerequisites**:
   - Create a PhysioNet account at [https://physionet.org/register/](https://physionet.org/register/)
   - Complete the required training course on human research data
   - Request access to the MIMIC dataset at:
     - MIMIC-III: [https://physionet.org/content/mimiciii/](https://physionet.org/content/mimiciii/)
     - MIMIC-IV: [https://physionet.org/content/mimiciv/](https://physionet.org/content/mimiciv/)

2. **Downloading the dataset**:

   After your credentials are approved, you can download the dataset using wget (if allowed by PhysioNet):

   ```bash
   # For MIMIC-III
   wget -r -N -c -np --user your-username --ask-password https://physionet.org/files/mimiciii/1.4/

   # For MIMIC-IV
   wget -r -N -c -np --user your-username --ask-password https://physionet.org/files/mimiciv/2.2/
   ```

   Alternatively, you can manually download the files from the PhysioNet website.

3. **Extract the data**:

   ```bash
   # Create a directory for the dataset
   mkdir -p /path/to/mimic/data

   # Extract the downloaded files
   tar -xzf downloaded_mimic_files.tar.gz -C /path/to/mimic/data
   ```

4. **Update your configuration**:

   Set the correct dataset path in your configuration file:

   ```yaml
   # config/training_config.yaml
   data:
     # Choose dataset version: "mimic-iii" or "mimic-iv"
     dataset_version: "mimic-iv"
     dataset_path: "/path/to/mimic/data"
   ```

### Preparing MIMIC Dataset

This project supports both MIMIC-III and MIMIC-IV datasets for training a medical code prediction model. 
After downloading the dataset as described above, you need to prepare it for training.

The project now supports three methods for dataset preparation:
1. **CSV-based processing** (original method)
2. **DuckDB-based processing** (recommended method)
3. **Tokenized data processing** (new, fastest method for training)

Using DuckDB offers several advantages:
- **Faster processing**: DuckDB provides significant performance improvements for large datasets
- **Reduced memory usage**: Data is processed directly in the database, reducing RAM requirements
- **Improved data management**: Data is stored in a structured database format
- **Efficient querying**: Allows for efficient data filtering and transformation

Using tokenized data processing offers additional advantages:
- **Fastest training**: Eliminates tokenization overhead during training
- **Reduced CPU usage**: No need to tokenize data on the fly
- **Consistent tokenization**: Ensures consistent tokenization across training runs
- **Efficient storage**: Stores only the tokenized data needed for training

#### MIMIC-III Dataset Preparation

1. **Required Tables**: The following MIMIC-III tables are used:
   - `NOTEEVENTS.csv`: Contains clinical notes
   - `DIAGNOSES_ICD.csv`: Contains diagnosis codes
   - `PROCEDURES_ICD.csv`: Contains procedure codes
   - `PATIENTS.csv`: Contains patient demographics

2. **Dataset Preparation**: Use our provided `just` command to prepare the dataset:

```shell script
# Using DuckDB (recommended)
just prepare-mimic path/to/mimic-iii/directory --use-duckdb

# Using CSV files (original method)
just prepare-mimic path/to/mimic-iii/directory
```

#### MIMIC-IV Dataset Preparation

1. **Required Tables**: The following MIMIC-IV tables are used:
   - `note.csv`: Contains clinical notes
   - `diagnoses_icd.csv`: Contains diagnosis codes
   - `procedures_icd.csv`: Contains procedure codes
   - `patients.csv`: Contains patient demographics

2. **Dataset Preparation**: Use our provided `just` command to prepare the dataset:

```shell script
# Using DuckDB (recommended)
just prepare-mimic path/to/mimic-iv/directory --dataset-version mimic-iv --use-duckdb

# Using CSV files (original method)
just prepare-mimic path/to/mimic-iv/directory --dataset-version mimic-iv
```

### Preparing Tokenized Data

After preparing the MIMIC dataset using DuckDB, the next required step is to pre-tokenize the data. This step is now mandatory for training, as it eliminates the tokenization overhead during training and significantly speeds up the training process.

To prepare tokenized data, use the `prepare_tokens.py` script:

```bash
python scripts/data_processing/prepare_tokens.py \
  --config config/training-config.yaml \
  --input-db data/mimic-dataset.duckdb \
  --output-dir data/tokens \
  --dataset-version mimic-iii \
  --verbose
```

Command-line options:

- `--config`: Path to the training configuration file
- `--input-db`: Path to the input DuckDB database created by prepare_mimic.py
- `--output-dir`: Directory to save the output DuckDB databases
- `--dataset-version`: Version of the MIMIC dataset to process
- `--verbose`: Enable verbose logging

This script will:
1. Load the train, validation, and test splits from the DuckDB database
2. Apply the same preprocessing logic as used during training
3. Save the preprocessed data into a single DuckDB file with separate tables for train, validation, and test

The output will be a single DuckDB database in the specified output directory:
- `mimic-iii-tokens.duckdb`: Contains three tables for tokenized training, validation, and test data

### Complete Data Preparation Workflow

To summarize, the complete workflow for preparing data for training is:

1. **Prepare the MIMIC dataset**:
   ```bash
   # For MIMIC-III
   just prepare-mimic path/to/mimic-iii/directory --use-duckdb
   
   # For MIMIC-IV
   just prepare-mimic path/to/mimic-iv/directory --dataset-version mimic-iv --use-duckdb
   ```

2. **Prepare the tokenized data**:
   ```bash
   python scripts/data_processing/prepare_tokens.py \
     --config config/training-config.yaml \
     --input-db data/mimic-dataset.duckdb \
     --output-dir data/tokens \
     --dataset-version mimic-iii
   ```

3. **Train the model** (using the tokenized data):
   ```bash
   python scripts/training/train.py \
     --config config/training-config.yaml \
     --tokens-dir data/tokens
   ```

This workflow ensures that the data is properly prepared and tokenized before training, resulting in faster and more efficient training.

### Training

#### Downloading the Base Model

Before training, you need to download the base model. The project uses DeepSeek models by default.

1. **Using Python**:

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch

   # Set cache directory (same as used by the project)
   cache_dir = "~/.cache/ft_test/models/base_models"

   # Download and cache the model
   model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
   model = AutoModelForCausalLM.from_pretrained(
       model_name, 
       torch_dtype=torch.float16,
       trust_remote_code=True,
       cache_dir=cache_dir
   )

   print(f"Model and tokenizer downloaded and cached to {cache_dir}")
   ```

2. **Alternatively, use the predict.py script with a dummy input**:

   ```bash
   python scripts/inference/predict.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --text "This is a test"
   ```

#### Training Configuration

The training configuration is specified in a YAML file. The default configuration file is `config/training-config.yaml`. You can modify this file or create a new one to customize the training process.

Example configuration:

```yaml
# Model configuration
model:
  model_name: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  max_length: 512

# Data configuration
data:
  dataset_version: "mimic-iii"
  dataset_path: "data/mimic-dataset.duckdb"
  tokens_dir: "data/tokens"

# Training configuration
training:
  output_dir: "output/deepseek-r1-distill-qwen-1.5b-lora"
  device: "cuda"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  save_checkpoint_epochs: 1
  checkpoint_dir: "output/checkpoints"
  logging_steps: 10
  evaluation_strategy: "epoch"
  save_strategy: "epoch"
  load_best_model_at_end: true
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  seed: 42

# LoRA configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

Note the data configuration options:
- `tokens_dir`: Directory containing the tokenized datasets (required for training)
- `dataset_path`: Path to the DuckDB database file (used by prepare_tokens.py)

#### Running the Training Script

To train the model, use the `train.py` script:

```bash
python scripts/training/train.py \
  --config config/training-config.yaml \
  --tokens-dir data/tokens
```

Command-line options:

- `--config`: Path to the training configuration file
- `--output-dir`: Output directory for the trained model (overrides config)
- `--data-path`: Path to the DuckDB database file (used by prepare_tokens.py, overrides config)
- `--dataset-version`: Version of the MIMIC dataset (overrides config)
- `--tokens-dir`: Path to the directory containing tokenized datasets (overrides config)
- `--device`: Device to run training on (cuda or cpu, overrides config)
- `--save-checkpoint-epochs`: Number of epochs after which to save checkpoints (overrides config)
- `--resume-from-checkpoint`: Path to a checkpoint to resume training from
- `--debug`: Enable debug mode with more verbose logging

#### Fine-tuning Options

- `--config`: Path to the training configuration file
- `--data`: Path to the data file or DuckDB database (overrides config)
- `--use-duckdb`: Use DuckDB for data loading (overrides config)
- `--output`: Output directory for the model (overrides config)
- `--epochs`: Number of training epochs (overrides config)
- `--batch-size`: Batch size for training (overrides config)
- `--device`: Device to run training on (cuda or cpu, overrides config)
- `--save-checkpoint-epochs`: Number of epochs after which to save checkpoints (overrides config)
- `--resume-from-checkpoint`: Path to a checkpoint to resume training from
- `--debug`: Enable debug mode with more verbose logging

#### Checkpoint Saving and Loading

The training process supports saving checkpoints at regular intervals and resuming training from a checkpoint:

- **Saving checkpoints**: Set `save_checkpoint_epochs` in the configuration file or use the `--save-checkpoint-epochs` command-line argument to specify the number of epochs after which to save checkpoints.
- **Resuming from a checkpoint**: Use the `--resume-from-checkpoint` command-line argument to specify the path to a checkpoint to resume training from.

### Inference

#### Using the Predict Script

The project includes a prediction script that can be used to generate predictions for a given text:

```bash
python scripts/inference/predict.py \
  --model output/deepseek-r1-distill-qwen-1.5b-lora \
  --text "Patient presents with chest pain radiating to the left arm. History of hypertension and diabetes."
```

Command-line options:

- `--model`: Path to the fine-tuned model or name of a Hugging Face model
- `--text`: Text to generate predictions for
- `--max-length`: Maximum length of the generated text
- `--device`: Device to run inference on (cuda or cpu)
- `--use-lora`: Whether to use LoRA for inference (required for fine-tuned models)
- `--output`: Path to save the predictions to (optional)

#### Using the API

The project includes a REST API for serving predictions:

1. **Start the API server**:

   ```bash
   python scripts/api/app.py --model output/deepseek-r1-distill-qwen-1.5b-lora --port 8000
   ```

2. **Make a prediction request**:

   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Patient presents with chest pain radiating to the left arm. History of hypertension and diabetes."}'
   ```

### Exporting the Model

The project includes a script for exporting the fine-tuned model to ONNX format:

```bash
python scripts/export/export_onnx.py \
  --model output/deepseek-r1-distill-qwen-1.5b-lora \
  --output output/onnx
```

Command-line options:

- `--model`: Path to the fine-tuned model
- `--output`: Path to save the exported model to
- `--device`: Device to run export on (cuda or cpu)
- `--use-lora`: Whether to use LoRA for export (required for fine-tuned models)

## Project Structure

```
ft_test/
├── config/                  # Configuration files
├── data/                    # Data files
├── output/                  # Output files
├── scripts/                 # Scripts
│   ├── api/                 # API scripts
│   ├── data_processing/     # Data processing scripts
│   ├── export/              # Export scripts
│   ├── inference/           # Inference scripts
│   └── training/            # Training scripts
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── model/               # Model definition and training
│   └── utils/               # Utility functions
├── tests/                   # Tests
├── README.md                # This file
└── pyproject.toml           # Project metadata
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The MIMIC-III and MIMIC-IV datasets are provided by PhysioNet
- The DeepSeek models are provided by DeepSeek AI
- The LoRA implementation is based on the PEFT library by Hugging Face