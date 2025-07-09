#!/usr/bin/env python
"""
Test script for model exporting functionality.

This script tests the model exporting functionality by exporting a model to LM Studio (GGUF) format.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.export.exporter import export_model
from src.utils.logging_utils import setup_logger

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test model exporting functionality to LM Studio (GGUF) format"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (with LoRA adapters)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level=logging.getLevelName(log_level))
    logger = logging.getLogger(__name__)

    logger.info(f"Testing model export functionality for model: {args.model_path}")

    try:
        # Test exporting to LM Studio format
        logger.info("Testing export to LM Studio (GGUF) format...")
        lmstudio_result = export_model(
            model_path=args.model_path,
            formats=["lmstudio"],
            quantization="q4_k_m"
        )
        logger.info(f"LM Studio export result: {lmstudio_result}")

        logger.info("Test completed successfully!")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
