#!/usr/bin/env python
"""
Inference script for predicting medical codes using the fine-tuned DeepSeek model.

This script loads the model and performs inference on text or PDF files.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))

# Import project modules
from src.model.inference import MedicalModelInference, load_model_from_config
from src.data.loader_duckdb import extract_text_from_pdf
from src.utils.logging_utils import setup_logger

# Configure logging
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Predict medical codes from text or PDF")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/inference-config.yaml",
        help="Path to the inference configuration file"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to the model directory (overrides config)"
    )
    parser.add_argument(
        "--text", 
        type=str, 
        default=None,
        help="Text to predict codes for"
    )
    parser.add_argument(
        "--pdf", 
        type=str, 
        default=None,
        help="Path to a PDF file to predict codes for"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save the prediction results (if not provided, prints to stdout)"
    )
    parser.add_argument(
        "--max-codes", 
        type=int, 
        default=10,
        help="Maximum number of codes to return"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default=None,
        help="Device to run inference on (cuda or cpu, overrides config)"
    )
    parser.add_argument(
        "--cache-dir", 
        type=str, 
        default=None,
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )
    parser.add_argument(
        "--use-litellm", 
        action="store_true",
        help="Use LiteLLM for inference instead of Transformers"
    )
    parser.add_argument(
        "--local-files-only", 
        action="store_true",
        help="Use only local files (don't download models)"
    )

    return parser.parse_args()

def predict_from_text(model: MedicalModelInference, text: str, max_codes: int = 10, use_litellm: bool = False) -> dict[str, Any]:
    """
    Predict medical codes from text.

    Args:
        model: Inference model
        text: Text to predict codes for
        max_codes: Maximum number of codes to return
        use_litellm: Whether to use LiteLLM for inference

    Returns:
        Dictionary containing the prediction results
    """
    logger.info(f"Predicting codes from text (length: {len(text)})")

    # Perform prediction
    if use_litellm:
        result = model.predict_with_litellm(text)
    else:
        result = model.predict_with_transformers(text)

    # Limit the number of codes
    if "codes" in result:
        result["codes"] = result["codes"][:max_codes]

    return result

def predict_from_pdf(model: MedicalModelInference, pdf_path: str | Path, max_codes: int = 10, use_litellm: bool = False) -> dict[str, Any]:
    """
    Predict medical codes from a PDF file.

    Args:
        model: Inference model
        pdf_path: Path to the PDF file
        max_codes: Maximum number of codes to return
        use_litellm: Whether to use LiteLLM for inference

    Returns:
        Dictionary containing the prediction results
    """
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting text from PDF: {pdf_path}")

    # Extract text from PDF
    pdf_data = extract_text_from_pdf(pdf_path)
    text = pdf_data["content"]

    if not text:
        logger.warning("No text could be extracted from the PDF")
        return {
            "error": "No text could be extracted from the PDF",
            "codes": [],
            "pdf_metadata": pdf_data["metadata"]
        }

    # Predict codes from the extracted text
    result = predict_from_text(model, text, max_codes, use_litellm)

    # Add PDF metadata to the result
    result["pdf_metadata"] = pdf_data["metadata"]

    return result

def save_results(results: dict[str, Any], output_path: str | Path | None = None) -> None:
    """
    Save or print the prediction results.

    Args:
        results: Prediction results
        output_path: Path to save the results (if None, prints to stdout)
    """
    # Convert results to JSON
    results_json = json.dumps(results, indent=2)

    if output_path:
        output_path = Path(output_path)
        # Create output directory if it doesn't exist
        output_dir = output_path.parent
        if str(output_dir) != "." and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(output_path, "w") as f:
            f.write(results_json)
        logger.info(f"Results saved to {output_path}")
    else:
        # Print to stdout
        print(results_json)

def main() -> None:
    """Main inference function."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(log_level=log_level)

    try:
        # Load the model
        if args.model:
            # Load model from command line argument
            logger.info(f"Loading model from {args.model}")
            model = MedicalModelInference(args.model, args.device, args.cache_dir, args.local_files_only)
            model.load_model()
        else:
            # Load model from config
            logger.info(f"Loading model from config {args.config}")
            model = load_model_from_config(args.config, args.cache_dir, args.local_files_only)

        # Check if we have text or PDF input
        if args.text:
            # Predict from text
            results = predict_from_text(model, args.text, args.max_codes, args.use_litellm)
        elif args.pdf:
            # Predict from PDF
            results = predict_from_pdf(model, args.pdf, args.max_codes, args.use_litellm)
        else:
            logger.error("No input provided. Use --text or --pdf to provide input.")
            sys.exit(1)

        # Save or print the results
        save_results(results, args.output)

    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
