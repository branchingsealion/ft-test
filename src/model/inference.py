"""
Model inference utilities for medical code prediction using the fine-tuned DeepSeek model.

This module provides functions and classes for performing inference with the
fine-tuned model to predict medical codes from text.

Supported backends:
- CUDA: For NVIDIA GPUs
- MPS: For Apple Silicon (M1/M2/M3)
- CPU: Fallback option when no hardware acceleration is available

Backend priority order: CUDA > MPS > ExecutorCH > CPU
"""

import logging
import time
from pathlib import Path
from typing import Any

import litellm
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# Default model cache directory
DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "ft_test" / "models"

# Backend-specific error messages
EXECUTORCH_IMPORT_ERROR = "ExecutorCH backend requested but torch_executorch module is not available. Install it with 'pip install torch_executorch'."
BACKEND_RUNTIME_ERROR = "Error initializing {} backend: {}. Falling back to CPU."

class MedicalModelInference:
    """
    Inference class for medical code prediction using the fine-tuned DeepSeek model.
    
    This class supports multiple hardware acceleration backends:
    
    - CUDA: For NVIDIA GPUs, requires PyTorch with CUDA support and NVIDIA drivers
    - MPS: For Apple Silicon (M1/M2/M3), requires PyTorch with MPS support
    - ExecutorCH: For optimized model execution, requires torch_executorch package
    - CPU: Fallback option when no hardware acceleration is available
    
    The backend is selected based on availability in the following priority order:
    CUDA > MPS > ExecutorCH > CPU
    
    You can explicitly specify a backend using the 'device' parameter. If the specified
    backend is not available, the class will automatically fall back to the best available option.
    
    Requirements for specific backends:
    - ExecutorCH: The torch_executorch package must be installed
    
    Note: When using ExecutorCH backend, ensure that your model is compatible
    with this backend. Some model architectures or operations may not be supported.
    """

    def __init__(self, model_path: str | Path, device: str | None = None, cache_dir: str | Path | None = None, local_files_only: bool = False):
        """
        Initialize the inference class with the fine-tuned model.

        Args:
            model_path: Path to the fine-tuned model
            device: Device to run inference on (cuda, mps, executor_ch, or cpu)
            cache_dir: Directory to cache downloaded models (if None, uses DEFAULT_MODEL_CACHE_DIR)
            local_files_only: Whether to only use local files (don't download models)
        """
        self.model_path = Path(model_path)
        
        # If device is explicitly specified, validate it
        if device:
            # Validate that the specified device is supported
            if device not in ["cuda", "mps", "cpu"]:
                logger.warning(f"Unsupported device '{device}' specified, will determine best available device")
                device = None
            # Check if CUDA is available when requested
            elif device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Will determine best available device.")
                device = None
            # Check if MPS is available when requested
            elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                logger.warning("MPS (Apple Silicon) requested but not available. Will determine best available device.")
                device = None
        
        # Set device to best available if not specified or if specified but not available
        self.device = device or self._get_best_available_device()
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_MODEL_CACHE_DIR
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing inference with model from: {self.model_path}")
        
        # Log device-specific information
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
            logger.info(f"Using CUDA device: {gpu_name}")
        elif self.device == "mps":
            logger.info("Using Apple Silicon (MPS) for inference")
        else:
            logger.info("Using CPU for inference")
            
        logger.info(f"Using model cache directory: {self.cache_dir}")

    def _get_best_available_device(self) -> str:
        """
        Determine the best available device for inference.

        Returns:
            String representing the best available device: 'cuda', 'mps', or 'cpu'
        """
        if torch.cuda.is_available():
            logger.info("CUDA is available, using GPU for inference")
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS is available, using Apple Silicon for inference")
            return "mps"
        else:
            logger.info("No hardware acceleration available, using CPU for inference")
            return "cpu"

    def load_model(self):
        """
        Load the fine-tuned model and tokenizer.
        """
        logger.info(f"Loading model from: {self.model_path}")

        try:
            # Load the configuration
            config = PeftConfig.from_pretrained(str(self.model_path))

            # Load the base model with caching
            logger.info(f"Loading base model: {config.base_model_name_or_path}")

            # Handle device mapping based on the selected device
            if self.device == "cuda":
                device_map = "auto"  # Let CUDA handle the device mapping automatically
            else:
                device_map = None  # For MPS and CPU, we'll move the model manually

            self.model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir / "base_models"),
                local_files_only=self.local_files_only
            )

            # If using MPS or CPU and device_map was None, move the model to the device
            if device_map is None:
                self.model = self.model.to(self.device)

            # Load the tokenizer with caching
            logger.info(f"Loading tokenizer: {config.base_model_name_or_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.base_model_name_or_path,
                trust_remote_code=True,
                cache_dir=str(self.cache_dir / "tokenizers"),
                local_files_only=self.local_files_only
            )

            # Ensure the tokenizer has padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load the LoRA adapter
            logger.info(f"Loading LoRA adapter: {self.model_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.model_path,
                torch_dtype=torch.float16,
                local_files_only=self.local_files_only
            )

            # If using MPS or CPU and we loaded the LoRA adapter, ensure it's on the right device
            if device_map is None:
                self.model = self.model.to(self.device)

            logger.info(f"Model and tokenizer loaded successfully on {self.device} backend")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def predict_with_transformers(self, text: str, max_length: int = 512) -> dict[str, Any]:
        """
        Predict medical codes using the Transformers pipeline.

        Args:
            text: Medical text to predict codes for
            max_length: Maximum length for generation

        Returns:
            Dictionary containing the predicted codes and metadata
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded")

        logger.info(f"Predicting codes for text (length: {len(text)})")

        try:
            start_time = time.time()

            # Format the input with instruction
            instruction = "Predict the medical codes for the following text:"
            formatted_input = f"{instruction}\n\n{text}"

            # Tokenize the input
            inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)

            # Generate the output
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    temperature=0.1,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode the output
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract the codes from the output (assuming they come after the instruction and input)
            response_text = decoded_output.replace(formatted_input, "").strip()

            # Parse the codes (assuming they are comma-separated)
            codes = [code.strip() for code in response_text.split(",") if code.strip()]

            processing_time = time.time() - start_time

            # Create the response
            response = {
                "codes": codes,
                "model_version": self.model_path.name,
                "processing_time": processing_time
            }

            logger.info(f"Prediction completed in {processing_time:.2f}s. Found {len(codes)} codes.")
            return response

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_with_litellm(self, text: str, max_tokens: int = 256) -> dict[str, Any]:
        """
        Predict medical codes using LiteLLM for inference.

        Args:
            text: Medical text to predict codes for
            max_tokens: Maximum tokens for generation

        Returns:
            Dictionary containing the predicted codes and metadata
        """
        logger.info(f"Predicting codes with LiteLLM for text (length: {len(text)})")

        try:
            start_time = time.time()

            # Format the input with instruction
            instruction = "Predict the medical codes for the following text:"
            formatted_input = f"{instruction}\n\n{text}"

            # Configure LiteLLM to use the local model
            model_name = "deepseek-ai/deepseek-coder-7b-chat"  # This should match the base model

            # Make the prediction
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": formatted_input}],
                max_tokens=max_tokens,
                temperature=0.1,
                local_model_path=self.model_path
            )

            # Extract the response text
            response_text = response.choices[0].message.content.strip()

            # Parse the codes (assuming they are comma-separated)
            codes = [code.strip() for code in response_text.split(",") if code.strip()]

            processing_time = time.time() - start_time

            # Create the response
            result = {
                "codes": codes,
                "model_version": self.model_path.name,
                "processing_time": processing_time
            }

            logger.info(f"Prediction completed in {processing_time:.2f}s. Found {len(codes)} codes.")
            return result

        except Exception as e:
            logger.error(f"Error during LiteLLM prediction: {str(e)}")
            raise

    def parse_codes(self, text: str) -> list[str]:
        """
        Parse medical codes from model output text.

        Args:
            text: Model output text containing codes

        Returns:
            List of parsed medical codes
        """
        # Basic parsing - split by commas and clean up
        codes = [code.strip() for code in text.split(",")]

        # Filter out empty strings and non-code text
        valid_codes = []
        for code in codes:
            if code and (":" in code or "-" in code):  # Simple heuristic for ICD/HCC codes
                valid_codes.append(code)

        return valid_codes

    def batch_predict(self, texts: list[str], batch_size: int = 4) -> list[dict[str, Any]]:
        """
        Perform batch prediction on multiple texts.

        Args:
            texts: List of medical texts to predict codes for
            batch_size: Batch size for prediction

        Returns:
            List of prediction results
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Call load_model() first.")
            raise ValueError("Model not loaded")

        logger.info(f"Batch predicting codes for {len(texts)} texts with batch size {batch_size}")

        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            batch_results = []
            for text in batch_texts:
                try:
                    result = self.predict_with_transformers(text)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing text in batch: {str(e)}")
                    batch_results.append({
                        "codes": [],
                        "error": str(e),
                        "model_version": self.model_path.name,
                        "processing_time": 0
                    })

            results.extend(batch_results)

        return results

def load_model_from_config(config_path: str | Path, cache_dir: str | Path | None = None, local_files_only: bool = False) -> MedicalModelInference:
    """
    Load a model from a configuration file.

    Args:
        config_path: Path to the configuration file
        cache_dir: Directory to cache downloaded models (if None, uses DEFAULT_MODEL_CACHE_DIR)
        local_files_only: Whether to only use local files (don't download models)

    Returns:
        Initialized MedicalModelInference
    """
    import yaml

    config_path = Path(config_path)
    logger.info(f"Loading inference configuration from {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        model_path = config.get("inference", {}).get("model_path", "")
        device = config.get("inference", {}).get("device", None)

        # Use cache_dir from config if not provided as argument
        if not cache_dir and "model_cache_dir" in config.get("inference", {}):
            cache_dir = config["inference"]["model_cache_dir"]

        if not model_path:
            raise ValueError("Model path not specified in configuration")

        inference = MedicalModelInference(model_path, device, cache_dir, local_files_only)
        inference.load_model()

        return inference

    except Exception as e:
        logger.error(f"Error loading inference configuration: {str(e)}")
        raise
