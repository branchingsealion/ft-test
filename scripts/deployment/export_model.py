#!/usr/bin/env python
"""
Script to export a fine-tuned model to GGUF format.

This script provides a command-line interface for exporting models to GGUF format.
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
        description="Export a fine-tuned model to GGUF format"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (with LoRA adapters)"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Directory to save exported GGUF model (defaults to model_path/exports/lmstudio)"
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Quantization method for GGUF conversion (q4_k_m, q5_k_m, q8_0, etc.)"
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
    setup_logger(log_level=log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Exporting model from: {args.model_path}")

    try:
        # Export the model
        result = export_model(
            model_path=args.model_path,
            output_dir=args.output_path,
            quantization=args.quantization
        )

        # Print results
        logger.info("Model export completed successfully!")
        logger.info(f"GGUF model exported to: {result}")

    except Exception as e:
        logger.error(f"Error exporting model: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
