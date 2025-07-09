"""
Unit tests for the API module.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the module to test
from src.deploy.api import app

# Create a test client
client = TestClient(app)

class TestAPI(unittest.TestCase):
    """Test cases for the API module."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the inference model and config
        self.mock_model = MagicMock()
        self.mock_config = {
            "inference": {
                "model_name": "test-model",
                "model_version": "v1.0",
                "code_descriptions": {
                    "ICD-10:E11.9": "Type 2 diabetes mellitus without complications",
                    "ICD-10:I10": "Essential (primary) hypertension"
                }
            }
        }

        # Set up mock prediction results
        self.mock_model.predict_with_transformers.return_value = {
            "codes": ["ICD-10:E11.9", "ICD-10:I10"],
            "model_version": "v1.0",
            "processing_time": 1.23
        }

        # Set up mock PDF extraction
        self.mock_pdf_data = {
            "content": "Patient has diabetes and hypertension.",
            "metadata": {
                "filename": "test.pdf",
                "page_count": 1,
                "extraction_time": 0.5
            },
            "sections": []
        }

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    def test_health_check(self, mock_get_config, mock_get_model):
        """Test the health check endpoint."""
        # Make a request to the health check endpoint
        response = client.get("/health")

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected fields
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    def test_model_info(self, mock_get_config, mock_get_model):
        """Test the model info endpoint."""
        # Set up the mock
        mock_get_config.return_value = self.mock_config

        # Make a request to the model info endpoint
        response = client.get("/info")

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected fields
        data = response.json()
        self.assertIn("model_name", data)
        self.assertEqual(data["model_name"], "test-model")
        self.assertIn("model_version", data)
        self.assertEqual(data["model_version"], "v1.0")
        self.assertIn("api_version", data)

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    def test_predict_codes(self, mock_get_config, mock_get_model):
        """Test the predict codes endpoint."""
        # Set up the mocks
        mock_get_model.return_value = self.mock_model
        mock_get_config.return_value = self.mock_config

        # Create a request payload
        payload = {
            "text": "Patient has diabetes and hypertension.",
            "max_codes": 10,
            "min_confidence": 0.5,
            "include_descriptions": False
        }

        # Make a request to the predict endpoint
        response = client.post("/predict", json=payload)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected fields
        data = response.json()
        self.assertIn("request_id", data)
        self.assertIn("timestamp", data)
        self.assertIn("predictions", data)
        self.assertIn("processing_time", data)
        self.assertIn("model_info", data)

        # Check that the model was called with the right arguments
        self.mock_model.predict_with_transformers.assert_called_once_with(payload["text"])

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    def test_predict_codes_with_descriptions(self, mock_get_config, mock_get_model):
        """Test the predict codes endpoint with descriptions."""
        # Set up the mocks
        mock_get_model.return_value = self.mock_model
        mock_get_config.return_value = self.mock_config

        # Create a request payload
        payload = {
            "text": "Patient has diabetes and hypertension.",
            "max_codes": 10,
            "min_confidence": 0.5,
            "include_descriptions": True
        }

        # Make a request to the predict endpoint
        response = client.post("/predict", json=payload)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)

        # Check that the response contains the expected fields
        data = response.json()
        self.assertIn("predictions", data)

        # Check that the predictions include descriptions
        predictions = data["predictions"]
        self.assertTrue(isinstance(predictions[0], dict))
        self.assertIn("code", predictions[0])
        self.assertIn("description", predictions[0])

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    @patch("src.deploy.api.extract_text_from_pdf")
    def test_predict_from_pdf(self, mock_extract_text, mock_get_config, mock_get_model):
        """Test the predict from PDF endpoint."""
        # Set up the mocks
        mock_get_model.return_value = self.mock_model
        mock_get_config.return_value = self.mock_config
        mock_extract_text.return_value = self.mock_pdf_data

        # Create a test PDF file
        test_file = Path("test.pdf")
        with open(test_file, "wb") as f:
            f.write(b"Test PDF content")

        try:
            # Create a multipart form request
            with open(test_file, "rb") as f:
                response = client.post(
                    "/predict/pdf",
                    files={"file": ("test.pdf", f, "application/pdf")},
                    data={
                        "max_codes": 10,
                        "min_confidence": 0.5,
                        "include_descriptions": False
                    }
                )

            # Check that the response is successful
            self.assertEqual(response.status_code, 200)

            # Check that the response contains the expected fields
            data = response.json()
            self.assertIn("request_id", data)
            self.assertIn("timestamp", data)
            self.assertIn("predictions", data)
            self.assertIn("processing_time", data)
            self.assertIn("model_info", data)

            # Check that the model was called with the right arguments
            self.mock_model.predict_with_transformers.assert_called_once_with(self.mock_pdf_data["content"])

        finally:
            # Clean up the test file
            if test_file.exists():
                test_file.unlink()

    @patch("src.deploy.api.get_inference_model")
    @patch("src.deploy.api.get_config")
    def test_error_handling(self, mock_get_config, mock_get_model):
        """Test error handling in the API."""
        # Set up the mocks to raise an exception
        mock_get_model.return_value = self.mock_model
        mock_get_config.return_value = self.mock_config
        self.mock_model.predict_with_transformers.side_effect = ValueError("Test error")

        # Create a request payload
        payload = {
            "text": "Patient has diabetes and hypertension.",
            "max_codes": 10,
            "min_confidence": 0.5,
            "include_descriptions": False
        }

        # Make a request to the predict endpoint
        response = client.post("/predict", json=payload)

        # Check that the response is an error
        self.assertEqual(response.status_code, 500)

        # Check that the response contains the expected error fields
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("error", data["detail"])
        self.assertIn("code", data["detail"]["error"])
        self.assertIn("message", data["detail"]["error"])
        self.assertIn("details", data["detail"]["error"])
        self.assertIn("request_id", data["detail"])
        self.assertIn("timestamp", data["detail"])

if __name__ == "__main__":
    unittest.main()
