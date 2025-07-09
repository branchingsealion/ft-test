"""
FastAPI REST API for medical code prediction.

This module provides a REST API for medical code prediction using the fine-tuned DeepSeek model.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from hypercorn.asyncio import serve
from hypercorn.config import Config
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.loader_duckdb import extract_text_from_pdf
# Import project modules
from src.model.inference import MedicalModelInference
from src.utils.logging_utils import setup_logger

# Configure logging
logger = logging.getLogger(__name__)

# Define API models
class PredictionRequest(BaseModel):
    """Request model for text prediction."""
    text: str = Field(..., description="Medical text to predict codes for")
    max_codes: int = Field(10, description="Maximum number of codes to return")
    min_confidence: float = Field(0.5, description="Minimum confidence threshold")
    include_descriptions: bool = Field(False, description="Include code descriptions")

class PredictionResponse(BaseModel):
    """Response model for prediction results."""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Timestamp of the request")
    predictions: list[dict[str, Any]] = Field(..., description="Predicted codes and metadata")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: dict[str, Any] = Field(..., description="Model information")

class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: dict[str, Any] = Field(..., description="Error details")
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: str = Field(..., description="Timestamp of the request")

# Create FastAPI app
app = FastAPI(
    title="Medical Code Prediction API",
    description="API for predicting medical codes from text using a fine-tuned DeepSeek model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_model = None
config = None

def get_inference_model() -> MedicalModelInference:
    """
    Get the inference model instance.

    Returns:
        MedicalModelInference instance
    """
    global inference_model
    if inference_model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    return inference_model

def get_config() -> dict[str, Any]:
    """
    Get the configuration.

    Returns:
        Configuration dictionary
    """
    global config
    if config is None:
        raise HTTPException(status_code=500, detail="Configuration not loaded")
    return config

@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the API on startup."""
    global inference_model, config

    try:
        # Set up logging
        setup_logger()

        # Load configuration
        config_path = Path(os.environ.get("CONFIG_PATH", "config/inference-config.yaml"))
        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load the model
        model_path = config.get("inference", {}).get("model_path", "")
        if not model_path:
            logger.error("Model path not specified in configuration")
            raise ValueError("Model path not specified in configuration")

        logger.info(f"Initializing inference model from {model_path}")
        inference_model = MedicalModelInference(model_path)
        inference_model.load_model()

        logger.info("API startup completed successfully")

    except Exception as e:
        logger.error(f"Error during API startup: {str(e)}")
        # We don't raise here to allow the API to start even if model loading fails
        # The endpoints will handle the None model case

@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    logger.info("API shutting down")
    # Add any cleanup code here if needed

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/info")
async def model_info(config: dict[str, Any] = Depends(get_config)) -> dict[str, str]:
    """Get model information."""
    return {
        "model_name": config.get("inference", {}).get("model_name", "unknown"),
        "model_version": config.get("inference", {}).get("model_version", "unknown"),
        "api_version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_codes(
    request: PredictionRequest,
    inference_model: MedicalModelInference = Depends(get_inference_model),
    config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    """
    Predict medical codes from text.

    Args:
        request: Prediction request with text and parameters

    Returns:
        Prediction response with codes and metadata
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"Processing prediction request {request_id} with {len(request.text)} characters")

    try:
        start_time = time.time()

        # Perform prediction
        result = inference_model.predict_with_transformers(request.text)

        # Filter codes based on request parameters
        codes = result.get("codes", [])[:request.max_codes]

        # Add code descriptions if requested
        if request.include_descriptions and "code_descriptions" in config.get("inference", {}):
            code_descriptions = config["inference"]["code_descriptions"]
            for i, code in enumerate(codes):
                if code in code_descriptions:
                    codes[i] = {
                        "code": code,
                        "description": code_descriptions[code]
                    }
                else:
                    codes[i] = {
                        "code": code,
                        "description": "Description not available"
                    }

        processing_time = time.time() - start_time

        # Create response
        response = {
            "request_id": request_id,
            "timestamp": timestamp,
            "predictions": codes,
            "processing_time": processing_time,
            "model_info": {
                "version": result.get("model_version", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"Completed prediction request {request_id} in {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing prediction request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "PREDICTION_ERROR",
                    "message": "Error predicting codes",
                    "details": {
                        "reason": str(e),
                        "suggestion": "Please try again with different text"
                    }
                },
                "request_id": request_id,
                "timestamp": timestamp
            }
        )

@app.post("/predict/pdf", response_model=PredictionResponse)
async def predict_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_codes: int = Form(10),
    min_confidence: float = Form(0.5),
    include_descriptions: bool = Form(False),
    inference_model: MedicalModelInference = Depends(get_inference_model),
    config: dict[str, Any] = Depends(get_config)
) -> dict[str, Any]:
    """
    Predict medical codes from a PDF file.

    Args:
        file: PDF file to extract text from
        max_codes: Maximum number of codes to return
        min_confidence: Minimum confidence threshold
        include_descriptions: Include code descriptions

    Returns:
        Prediction response with codes and metadata
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    logger.info(f"Processing PDF prediction request {request_id} with file {file.filename}")

    try:
        start_time = time.time()

        # Save the uploaded file temporarily
        temp_file_path = Path("/tmp") / f"{request_id}_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)

        # Schedule file cleanup
        background_tasks.add_task(os.remove, str(temp_file_path))

        # Extract text from PDF
        pdf_data = extract_text_from_pdf(temp_file_path)
        text = pdf_data["content"]

        if not text:
            raise ValueError("No text could be extracted from the PDF")

        # Perform prediction
        result = inference_model.predict_with_transformers(text)

        # Filter codes based on request parameters
        codes = result.get("codes", [])[:max_codes]

        # Add code descriptions if requested
        if include_descriptions and "code_descriptions" in config.get("inference", {}):
            code_descriptions = config["inference"]["code_descriptions"]
            for i, code in enumerate(codes):
                if code in code_descriptions:
                    codes[i] = {
                        "code": code,
                        "description": code_descriptions[code]
                    }
                else:
                    codes[i] = {
                        "code": code,
                        "description": "Description not available"
                    }

        processing_time = time.time() - start_time

        # Create response
        response = {
            "request_id": request_id,
            "timestamp": timestamp,
            "predictions": codes,
            "processing_time": processing_time,
            "model_info": {
                "version": result.get("model_version", "unknown"),
                "timestamp": datetime.now().isoformat()
            }
        }

        logger.info(f"Completed PDF prediction request {request_id} in {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing PDF prediction request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "PDF_PROCESSING_ERROR",
                    "message": "PDF file could not be processed",
                    "details": {
                        "reason": str(e),
                        "suggestion": "Please ensure the PDF is valid and contains text"
                    }
                },
                "request_id": request_id,
                "timestamp": timestamp
            }
        )

def start_api(host: str = "0.0.0.0", port: int = 8000, config_path: str | Path = "config/inference-config.yaml") -> None:
    """
    Start the FastAPI server.

    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        config_path: Path to the configuration file
    """
    # Set the configuration path environment variable
    os.environ["CONFIG_PATH"] = str(config_path)

    # Configure hypercorn
    hypercorn_config = Config()
    hypercorn_config.bind = [f"{host}:{port}"]
    hypercorn_config.use_reloader = False

    # Import asyncio to run the server
    import asyncio

    # Start the server
    asyncio.run(serve(app, hypercorn_config))

if __name__ == "__main__":
    # This is used when running the API directly with python -m src.deploy.api
    start_api()
