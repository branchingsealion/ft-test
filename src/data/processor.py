#!/usr/bin/env python
"""
Data preprocessing utilities for text datasets and medical code prediction.

This module provides:
1. A MedicalDataPreprocessor class for medical text and code preprocessing
2. Functions to preprocess data for causal language modeling:
   - Concatenate "prompt" and "completion" into a single string
   - Tokenize the concatenated string with a given AutoTokenizer
   - Add labels equal to input_ids for causal language modeling
3. Parallel processing utilities for efficient dataset processing

The module uses joblib for parallel processing and tqdm_joblib to show progress.
It ensures that no shared preprocessor or tokenizer is used.
"""

import logging
import re
from typing import Any, TypeVar

import torch
from transformers import AutoTokenizer

# Try to import datasets, but don't fail if it's not available
try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    # Create a type alias for type hints
    Dataset = TypeVar('Dataset')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalDataPreprocessor:
    """
    Preprocessor for medical text data and associated codes.
    """

    def __init__(self, model_name: str, max_length: int = 512):
        """
        Initialize the preprocessor with a tokenizer.

        Args:
            model_name: Name or path of the model for tokenization
            max_length: Maximum sequence length for tokenization
        """
        logger.info(f"Initializing preprocessor with model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Ensure the tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        # Basic preprocessing steps
        processed_text = text.strip()

        # Remove excessive whitespace
        processed_text = re.sub(r'\s+', ' ', processed_text)

        return processed_text

    def tokenize(self, text: str, padding: bool = True, truncation: bool = True) -> dict[str, torch.Tensor]:
        """
        Tokenize text for model input.

        Args:
            text: Text to tokenize
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences

        Returns:
            dictionary of tokenized inputs
        """
        logger.debug(f"Tokenizing text (length: {len(text)})")

        # Preprocess the text first
        processed_text = self.preprocess_text(text)

        # Tokenize the text
        inputs = self.tokenizer(
            processed_text,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return inputs

    def format_model_input(self, text: str, instruction: str = "Predict the medical codes for the following text:") -> str:
        """
        Format text for model input with instruction.

        Args:
            text: Medical text to format
            instruction: Instruction for the model

        Returns:
            Formatted text for model input
        """
        # Format the input according to the model's expected format
        # This example uses a simple instruction-based format
        formatted_input = f"{instruction}\n\n{text}"

        return formatted_input

    def format_codes_for_training(self, codes: list[str]) -> str:
        """
        Format codes for training the model.

        Args:
            codes: list of medical codes

        Returns:
            Formatted codes string
        """
        # Join codes with commas for training
        return ", ".join(codes)

    def create_prompt_completion_pair(self, example: dict[str, Any]) -> tuple[str, str]:
        """
        Create a prompt-completion pair for fine-tuning.

        Args:
            example: Example with text and codes

        Returns:
            tuple of (prompt, completion)
        """
        text = example["text"]
        codes = example["codes"]

        # Create the prompt with instruction
        prompt = self.format_model_input(text)

        # Create the completion with formatted codes
        completion = self.format_codes_for_training(codes)

        return prompt, completion

    def batch_create_prompt_completion_pairs(self, examples: list[dict[str, Any]]) -> list[dict[str, str]]:
        """
        Create prompt-completion pairs for a batch of examples.

        Args:
            examples: list of examples with text and codes

        Returns:
            list of dictionaries with prompt and completion
        """
        logger.info(f"Creating prompt-completion pairs for {len(examples)} examples")

        pairs = []
        for example in examples:
            prompt, completion = self.create_prompt_completion_pair(example)
            pairs.append({
                "prompt": prompt,
                "completion": completion
            })

        return pairs
        
    def preprocess_for_causal_lm(self, example: dict[str, str]) -> dict[str, torch.Tensor]:
        """
        Preprocess a single example for causal language modeling.
        
        This function implements the preprocessing pipeline required for training a
        PeftModelForCausalLM using Hugging Face Transformers and LoRA:
        1. Concatenates "prompt" and "completion" fields into a single string
        2. Tokenizes the concatenated string with the tokenizer
        3. Adds labels equal to input_ids for causal language modeling
        
        Args:
            example: dictionary with "prompt" and "completion" fields
            
        Returns:
            dictionary with "input_ids", "attention_mask", and "labels" fields
            suitable for passing to Hugging Face Trainer for causal language modeling
        """
        # Concatenate prompt and completion into a single string
        prompt = example["prompt"]
        completion = example["completion"]
        full_text = f"{prompt}{completion}"
        
        # Tokenize the concatenated string with the tokenizer
        tokenized = self.tokenizer(
            full_text,
            padding="max_length",  # Pad to max_length
            truncation=True,       # Truncate if longer than max_length
            max_length=self.max_length,
            return_tensors="pt"    # Return PyTorch tensors
        )
        
        # Add labels equal to input_ids for causal language modeling
        # For causal LM, the labels are the same as the input_ids
        result = {
            "input_ids": tokenized["input_ids"][0],  # Remove batch dimension
            "attention_mask": tokenized["attention_mask"][0],  # Remove batch dimension
            "labels": tokenized["input_ids"][0].clone()  # Labels are the same as input_ids for causal LM
        }
        
        return result
