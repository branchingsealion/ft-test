"""
Model training utilities for fine-tuning the DeepSeek 7B model using LoRA.

This module provides functions and classes for fine-tuning the DeepSeek model
on medical text data for code prediction.
"""

import datetime
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# Import model export functionality
from src.model.export.exporter import export_model

# Configure logging
logger = logging.getLogger(__name__)

class MedicalModelTrainer:
    """
    Trainer for fine-tuning DeepSeek model on medical code prediction.
    """

    def __init__(self, model_config: dict[str, Any], training_config: dict[str, Any]):
        """
        Initialize the trainer with model and training configurations.

        Args:
            model_config: Dictionary containing model configuration
            training_config: Dictionary containing training configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.tokenizer = None

        logger.info(f"Initializing trainer with model: {model_config['model_name']}")

        # Set device
        device_config = training_config.get("device", "auto")
        self.device = self._get_best_available_device() if device_config == "auto" else device_config
        logger.info(f"Using device: {self.device}")

        # Initialize memory tracking
        self.memory_stats = []

    @staticmethod
    def _get_best_available_device() -> str:
        """
        Determine the best available device for training.

        Returns:
            String representing the best available device: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU for training")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS is available, using Apple Silicon for training")
            return "mps"
        else:
            logger.info("No GPU acceleration available, using CPU for training")
            return "cpu"

    def _log_memory_usage(self, step: str) -> None:
        """
        Log memory usage statistics.

        Args:
            step: Current step description
        """
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
            gpu_utilization = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
        else:
            gpu_memory = 0
            gpu_utilization = 0

        import psutil
        ram_usage = psutil.virtual_memory().percent / 100

        memory_stat = {
            "gpu_utilized": gpu_utilization,
            "ram_utilized": ram_usage,
            "gpu_memory_gb": gpu_memory,
            "batch_size": self.model_config.get("batch_size", 0),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_name": self.model_config.get("model_name", ""),
            "step": step
        }

        self.memory_stats.append(memory_stat)
        logger.info(f"Memory usage - Step: {step}, GPU: {gpu_memory:.2f}GB ({gpu_utilization:.2%}), RAM: {ram_usage:.2%}")

    def load_model(self) -> None:
        """
        Load the base model and tokenizer.
        """
        logger.info(f"Loading model: {self.model_config['model_name']}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config["model_name"],
                trust_remote_code=True
            )

            # Ensure the tokenizer has padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Handle device mapping based on the selected device
            if self.device == "cuda":
                device_map = "auto"  # Let CUDA handle the device mapping automatically
            else:
                device_map = None  # For MPS and CPU, we'll move the model manually

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config["model_name"],
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True
            )

            # If using MPS or CPU and device_map was None, move the model to the device
            if device_map is None:
                self.model = self.model.to(self.device)

            self._log_memory_usage("model_loaded")
            logger.info("Model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def prepare_for_lora(self) -> None:
        """
        Prepare the model for LoRA fine-tuning.
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded")

        logger.info("Preparing model for LoRA fine-tuning")

        try:
            # Prepare model for k-bit training if needed
            self.model = prepare_model_for_kbit_training(self.model)

            # Configure LoRA
            lora_config = LoraConfig(
                r=self.model_config.get("lora_r", 8),
                lora_alpha=self.model_config.get("lora_alpha", 16),
                lora_dropout=self.model_config.get("lora_dropout", 0.05),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )

            # Apply LoRA to model
            self.model = get_peft_model(self.model, lora_config)

            # Ensure the model is on the correct device after applying LoRA
            # This is especially important for MPS devices
            if self.device != "cuda":  # For MPS and CPU
                self.model = self.model.to(self.device)

            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of {total_params:,} total)")

            self._log_memory_usage("lora_prepared")

        except Exception as e:
            logger.error(f"Error preparing model for LoRA: {str(e)}")
            raise

    def train(self, train_dataset: Any, eval_dataset: Any | None = None) -> dict[str, Any]:
        """
        Train the model using the provided datasets.
        
        This method supports checkpoint saving and loading:
        - Set 'save_checkpoint_epochs' in the training config to save checkpoints after specific number of epochs
        - Set 'checkpoint_dir' in the training config to specify where checkpoints are saved
        - Set 'resume_from_checkpoint' in the training config to resume training from a specific checkpoint
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)

        Returns:
            Dictionary containing training metrics
        """
        if self.model is None:
            logger.error("Model not prepared. Call load_model() and prepare_for_lora() first.")
            raise ValueError("Model not prepared")

        logger.info("Starting model training")

        try:
            # Set up training arguments
            output_dir = self.training_config.get("output_dir", "output")
            
            # Set up checkpoint directory
            checkpoint_dir = self.training_config.get("checkpoint_dir", "")
            if not checkpoint_dir:
                checkpoint_dir = str(Path(output_dir) / "checkpoints")
            
            # Determine if we're resuming from a checkpoint
            resume_from_checkpoint = self.training_config.get("resume_from_checkpoint", None)
            if resume_from_checkpoint:
                logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            
            # Get save checkpoint epochs
            save_checkpoint_epochs = int(self.training_config.get("save_checkpoint_epochs", 0))
            
            # Determine save strategy and save steps
            # If save_checkpoint_epochs is set, we'll use "epoch" strategy
            # Otherwise, we'll use "steps" strategy with the configured save_steps
            save_strategy = "epoch" if save_checkpoint_epochs > 0 else "steps"
            save_steps = int(self.training_config.get("save_steps", 100))
            
            # If using epoch-based saving, set save_total_limit to keep only the specified number of checkpoints
            # This prevents accumulating too many checkpoints
            save_total_limit = None
            if save_checkpoint_epochs > 0:
                # Save only the most recent checkpoints (one per save_checkpoint_epochs)
                total_epochs = int(self.training_config.get("epochs", 3))
                save_total_limit = (total_epochs // save_checkpoint_epochs) + 1
            
            training_args = TrainingArguments(
                output_dir=checkpoint_dir,  # Save checkpoints to checkpoint_dir
                num_train_epochs=float(self.training_config.get("epochs", 3)),
                per_device_train_batch_size=int(self.model_config.get("batch_size", 4)),
                gradient_accumulation_steps=int(self.model_config.get("gradient_accumulation_steps", 4)),
                learning_rate=float(self.model_config.get("learning_rate", 2e-4)),
                weight_decay=0.01,
                logging_dir=str(Path(output_dir) / "logs"),
                logging_steps=int(self.training_config.get("logging_steps", 10)),
                save_steps=save_steps,
                eval_steps=int(self.training_config.get("eval_steps", 50)) if eval_dataset else None,
                warmup_steps=int(self.training_config.get("warmup_steps", 100)),
                fp16=(self.device == "cuda"),
                report_to="none",  # Disable wandb, etc.
                eval_strategy="steps" if eval_dataset else "no",
                save_strategy=save_strategy,
                save_total_limit=save_total_limit,
                load_best_model_at_end=True if eval_dataset else False,
                remove_unused_columns=False,
                label_names=[]
            )

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )

            # Train the model
            self._log_memory_usage("training_start")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self._log_memory_usage("training_end")

            # Save the final model to the output directory (not the checkpoint directory)
            logger.info(f"Saving final model to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)

            # Collect metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # Add memory stats to metrics
            metrics["memory_stats"] = self.memory_stats

            logger.info(f"Training completed. Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self._log_memory_usage("training_error")
            raise

    def evaluate(self, eval_dataset: Any) -> dict[str, Any]:
        """
        Evaluate the model on the provided dataset.

        Args:
            eval_dataset: Evaluation dataset

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            logger.error("Model not prepared. Call load_model() and prepare_for_lora() first.")
            raise ValueError("Model not prepared")

        logger.info("Evaluating model")

        try:
            # Set up evaluation arguments
            output_dir = self.training_config.get("output_dir", "output")
            eval_args = TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=self.model_config.get("batch_size", 4),
                report_to="none"
            )

            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=eval_args,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )

            # Evaluate the model
            self._log_memory_usage("evaluation_start")
            metrics = trainer.evaluate()
            self._log_memory_usage("evaluation_end")

            # Log metrics
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            logger.info(f"Evaluation completed. Metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            self._log_memory_usage("evaluation_error")
            raise

    def export_model(self, formats: list[str] | None = None, output_dir: str | Path | None = None,
                    quantization: str = "q4_k_m") -> Path:
        """
        Export the trained model to different formats.

        Args:
            formats: list of formats to export to ("ollama", "lmstudio", or both)
            output_dir: Directory to save exported models (defaults to output_dir/exports)
            quantization: Quantization method for GGUF conversion

        Returns:
            Path to exported models
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            raise ValueError("Model not trained")

        # Get the output directory from training config if not specified
        if output_dir is None:
            output_dir = self.training_config.get("output_dir", "output")

        # Set default formats if not specified
        if formats is None:
            formats = ["lmstudio"]

        logger.info(f"Exporting model to formats: {formats}")

        try:
            # Export the model
            results = export_model(
                model_path=output_dir,
                output_dir=output_dir,
                quantization=quantization
            )

            logger.info("Model export completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            raise

    def metrics_format(self, metrics: dict[str, float]) -> dict[str, Any]:
        """
        Format metrics for display.

        Args:
            metrics: Dictionary of metrics to format

        Returns:
            Formatted metrics dictionary
        """
        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_runtime" in k:
                # Convert seconds to human-readable format
                metrics_copy[k] = str(datetime.timedelta(seconds=int(v)))
            elif k == "total_flos":
                # Convert FLOPS to GigaFLOPS
                metrics_copy[k] = f"{int(v) >> 30}GF"
            elif isinstance(v, float):
                # Round floats to 4 decimal places
                metrics_copy[k] = round(v, 4)

        return metrics_copy

    def log_metrics(self, split: str, metrics: dict[str, float]) -> None:
        """
        Log metrics in a formatted way.

        Args:
            split: Mode/split name: one of 'train', 'eval', 'test'
            metrics: Dictionary of metrics to log
        """
        print(f"***** {split} metrics *****")
        metrics_formatted = self.metrics_format(metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")

    def save_metrics(self, split: str, metrics: dict[str, float], combined: bool = True) -> None:
        """
        Save metrics into a json file for that split, e.g. 'train_results.json'.

        Args:
            split: Mode/split name: one of 'train', 'eval', 'test', 'all'
            metrics: Dictionary of metrics to save
            combined: Whether to create combined metrics by updating 'all_results.json'
        """
        output_dir = Path(self.training_config.get("output_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics for this split
        json_path = output_dir / f"{split}_results.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Update combined metrics if requested
        if combined:
            combined_path = output_dir / "all_results.json"
            if combined_path.exists():
                with open(combined_path, "r") as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}
                
            all_metrics.update(metrics)
            with open(combined_path, "w") as f:
                json.dump(all_metrics, f, indent=2)


def create_trainer_from_config(config_path: str | Path) -> MedicalModelTrainer:
    """
    Create a trainer from a configuration file.
    
    The configuration file should be a YAML file with the following structure:
    ```yaml
    model:
      # Model configuration parameters
      model_name: "model_name"
      # ...
    
    training:
      # Training configuration parameters
      epochs: 3
      save_steps: 100
      # ...
      
      # Checkpoint configuration parameters
      save_checkpoint_epochs: 1  # Save checkpoint after every epoch
      checkpoint_dir: ""  # If empty, will use a subdirectory of output_dir
      resume_from_checkpoint: ""  # Path to a checkpoint to resume training from
    ```

    Args:
        config_path: Path to the configuration file

    Returns:
        Initialized MedicalModelTrainer
    """
    import yaml

    logger.info(f"Loading configuration from {config_path}")

    try:
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_config = config.get("model", {})
        training_config = config.get("training", {})

        return MedicalModelTrainer(model_config, training_config)

    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
