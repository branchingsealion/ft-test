"""
Model exporter for packaging fine-tuned models in GGUF format.

This module provides functions for exporting models to GGUF format compatible with
LM Studio and other inference engines that support GGUF.
"""

import logging
import subprocess
from pathlib import Path
from typing import Tuple, Union

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

class ModelExporter:
    """
    Exporter for packaging fine-tuned models in different formats.
    """

    def __init__(self, model_path: Union[str, Path], output_dir: Union[str, Path] = None):
        """
        Initialize the model exporter.

        Args:
            model_path: Path to the fine-tuned model (with LoRA adapters)
            output_dir: Directory to save exported models (defaults to model_path/exports)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path / "exports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure the model path exists
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {self.model_path}")

        logger.info(f"Initialized model exporter for model: {self.model_path}")
        logger.info(f"Exports will be saved to: {self.output_dir}")

    def _load_model_and_tokenizer(self) -> Tuple[PeftModel, AutoTokenizer]:
        """
        Load the model and tokenizer.

        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model from: {self.model_path}")

        try:
            # Load the configuration
            config = PeftConfig.from_pretrained(self.model_path)

            # Load the base model
            logger.info(f"Loading base model: {config.base_model_name_or_path}")
            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # Load the tokenizer
            logger.info(f"Loading tokenizer: {config.base_model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=True
            )

            # Ensure the tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load the LoRA adapter
            logger.info(f"Loading LoRA adapter: {self.model_path}")
            model = PeftModel.from_pretrained(
                model,
                self.model_path,
                torch_dtype=torch.float16
            )

            logger.info("Model and tokenizer loaded successfully")
            return model, tokenizer

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _merge_and_save_model(self, output_path: Union[str, Path]) -> Path:
        """
        Merge LoRA adapters with the base model and save the result.

        Args:
            output_path: Path to save the merged model

        Returns:
            Path to the saved model
        """
        logger.info(f"Merging LoRA adapters with base model and saving to: {output_path}")

        try:
            # Load model and tokenizer
            model, tokenizer = self._load_model_and_tokenizer()

            # Merge LoRA adapters with base model
            logger.info("Merging LoRA adapters with base model")
            model = model.merge_and_unload()

            # Save the merged model
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving merged model to: {output_path}")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            logger.info("Model merged and saved successfully")
            return output_path

        except Exception as e:
            logger.error(f"Error merging and saving model: {str(e)}")
            raise


    def export_to_lmstudio(self, quantization: str = "q4_k_m") -> Path:
        """
        Export the model to LM Studio format (GGUF).

        Args:
            quantization: Quantization method to use (q4_k_m, q5_k_m, q8_0, etc.)

        Returns:
            Path to the exported GGUF model file
        """
        logger.info(f"Exporting model to LM Studio format with quantization: {quantization}")

        try:
            # Create output directory for LM Studio model
            lmstudio_dir = self.output_dir / "lmstudio"
            lmstudio_dir.mkdir(parents=True, exist_ok=True)

            # Merge LoRA adapters with base model and save
            merged_model_path = self.output_dir / "merged_model"
            self._merge_and_save_model(merged_model_path)

            # Check if llama.cpp is available
            try:
                subprocess.run(["python", "-m", "llama_cpp.server"], capture_output=True, text=True, check=False)
                llama_cpp_available = True
            except (subprocess.SubprocessError, ModuleNotFoundError):
                llama_cpp_available = False

            if llama_cpp_available:
                # Use llama-cpp-python to convert to GGUF
                logger.info("Converting model to GGUF format using llama-cpp-python")

                # Output GGUF file path
                gguf_path = lmstudio_dir / f"{self.model_path.name}_{quantization}.gguf"

                # Run conversion script
                cmd = [
                    "python", "-m", "llama_cpp.convert_hf_to_gguf",
                    "--outfile", str(gguf_path),
                    "--outtype", quantization,
                    str(merged_model_path)
                ]

                logger.info(f"Running conversion command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"Conversion output: {result.stdout}")

            else:
                # Provide instructions for manual conversion
                logger.warning("llama-cpp-python not available for automatic GGUF conversion")
                logger.info("Providing instructions for manual conversion instead")

                # Create a placeholder GGUF file
                gguf_path = lmstudio_dir / f"{self.model_path.name}_{quantization}.gguf.placeholder"
                with open(gguf_path, "w") as f:
                    f.write("This is a placeholder file. Follow the instructions in the README to convert the model to GGUF format.")

            # Create README with instructions
            readme_path = lmstudio_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(f"# {self.model_path.name} - LM Studio Model\n\n")
                f.write("This directory contains the model exported for use with LM Studio.\n\n")

                if llama_cpp_available:
                    f.write("## Usage\n\n")
                    f.write("To use this model with LM Studio:\n\n")
                    f.write("1. Download and install LM Studio from https://lmstudio.ai\n")
                    f.write(f"2. Open LM Studio and import the GGUF file: {gguf_path.name}\n")
                else:
                    f.write("## Manual Conversion\n\n")
                    f.write("To convert this model to GGUF format for use with LM Studio:\n\n")
                    f.write("1. Install llama-cpp-python:\n\n")
                    f.write("```bash\n")
                    f.write("pip install llama-cpp-python\n")
                    f.write("```\n\n")
                    f.write("2. Convert the model to GGUF format:\n\n")
                    f.write("```bash\n")
                    f.write(f"python -m llama_cpp.convert_hf_to_gguf --outfile {self.model_path.name}_{quantization}.gguf --outtype {quantization} {merged_model_path}\n")
                    f.write("```\n\n")
                    f.write("3. Download and install LM Studio from https://lmstudio.ai\n")
                    f.write(f"4. Open LM Studio and import the GGUF file\n")

            logger.info(f"Created README with instructions at: {readme_path}")
            logger.info(f"LM Studio model exported to: {lmstudio_dir}")

            return gguf_path

        except Exception as e:
            logger.error(f"Error exporting model to LM Studio format: {str(e)}")
            raise

def export_model(model_path: Union[str, Path], output_dir: Union[str, Path] = None, 
                quantization: str = "q4_k_m") -> Path:
    """
    Export a model to GGUF format.

    Args:
        model_path: Path to the fine-tuned model
        output_dir: Directory to save exported GGUF model
        quantization: Quantization method for GGUF conversion

    Returns:
        Path to the exported GGUF model file
    """
    # Create exporter
    exporter = ModelExporter(model_path, output_dir)

    # Export to GGUF format
    logger.info(f"Exporting model to GGUF format")
    result = exporter.export_to_lmstudio(quantization)

    return result
