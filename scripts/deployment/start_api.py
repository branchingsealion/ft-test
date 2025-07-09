#!/usr/bin/env python
"""
Script to start the FastAPI server for medical code prediction.

This script loads the configuration and starts the API server.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parents[1].absolute()))

# Import project modules
from src.deploy.api import start_api
from src.utils.logging_utils import setup_logger

# Configure logging
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the medical code prediction API server")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/inference-config.yaml",
        help="Path to the inference configuration file"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default=None,
        help="Host to bind the server to (overrides config)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Port to bind the server to (overrides config)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode with more verbose logging"
    )

    return parser.parse_args()

def main():
    """Main function to start the API server."""
    # Parse command line arguments
    args = parse_args()

    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger(log_level=log_level)

    try:
        # Load configuration
        config_path = args.config
        logger.info(f"Loading configuration from {config_path}")

        # Get host and port from command line arguments or config
        host = args.host
        port = args.port

        if not host or not port:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            api_config = config.get("api", {})
            host = host or api_config.get("host", "0.0.0.0")
            port = port or api_config.get("port", 8000)

        # Start the API server
        logger.info(f"Starting API server on {host}:{port}")
        start_api(host=host, port=port, config_path=config_path)

    except Exception as e:
        logger.error(f"Error starting API server: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
