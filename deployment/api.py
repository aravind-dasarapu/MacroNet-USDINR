"""
FastAPI deployment module for FX prediction model.

Provides REST API endpoints for model inference with input validation,
preprocessing, and prediction logging.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.config_loader import get_config
from src.logger import setup_logger
from src.model_training import DirectionalLogCoshLoss


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    features: List[List[float]] = Field(
        ...,
        description="Time series features as a 2D array (timesteps x features)",
        example=[[0.1, 0.2, 0.3], [0.15, 0.25, 0.35]],
    )
    window_size: Optional[int] = Field(
        None, description="Expected window size (auto-detected if not provided)"
    )


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: float
    direction: str
    confidence: float
    timestamp: str
    model_version: str


class ModelServer:
    """Handles model loading and inference serving."""

    def __init__(self, config_path: str = None):

        self.base_dir = Path(__file__).resolve().parent

        self.config = get_config(config_path)
        self.logger = setup_logger("model_server")

        raw_model_dir = self.config.deployment["model_dir"]
        self.model_dir = (self.base_dir / raw_model_dir).resolve()

        self.model = None
        self.model_version = None
        self.window_size = self.config.model["window_size"]

        self._load_latest_model()

    def _load_latest_model(self):
        """Load the most recent trained model."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Find latest model file
        model_files = list(self.model_dir.glob("*_final.keras"))

        if not model_files:
            raise FileNotFoundError(
                f"No trained models found in {self.model_dir}. "
                "Train a model first using model_training.py"
            )

        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)

        self.logger.info(f"Loading model: {latest_model}")
        self.model = tf.keras.models.load_model(
            latest_model,
            custom_objects={"DirectionalLogCoshLoss": DirectionalLogCoshLoss},
        )
        self.model_version = latest_model.stem

        self.logger.info(f"Model loaded successfully: {self.model_version}")

    def validate_input(self, features: List[List[float]]) -> np.ndarray:
        """
        Validate and convert input features to numpy array.

        Args:
            features: 2D list of features

        Returns:
            Validated numpy array
        """
        features_array = np.array(features, dtype="float32")

        # Check shape
        if features_array.ndim != 2:
            raise ValueError(
                f"Features must be 2D array, got shape: {features_array.shape}"
            )

        timesteps, num_features = features_array.shape

        if timesteps != self.window_size:
            raise ValueError(f"Expected {self.window_size} timesteps, got {timesteps}")

        # Reshape for model input: (1, timesteps, features)
        features_array = features_array.reshape(1, timesteps, num_features)

        return features_array

    def predict(self, features: np.ndarray) -> Dict:
        """
        Generate prediction from features.

        Args:
            features: Preprocessed feature array

        Returns:
            Dictionary with prediction results
        """
        # Generate prediction
        prediction = self.model.predict(features, verbose=0)[0][0]

        # Determine direction
        direction = "UP" if prediction > 0 else "DOWN"

        # Calculate confidence (normalized absolute value)
        confidence = min(abs(prediction) / 0.1, 1.0)

        return {
            "prediction": float(prediction),
            "direction": direction,
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
        }


# Initialize FastAPI app
app = FastAPI(
    title="FX Prediction API",
    description="USD/INR price direction prediction using deep learning",
    version="1.0.4",
)

# Configure CORS

config = get_config()
model_server = ModelServer()

if config.deployment["enable_cors"]:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize model server


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "FX Prediction API",
        "version": "1.0.0",
        "model_version": model_server.model_version,
        "endpoints": {
            "/predict": "POST - Generate prediction",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model_server.model is not None,
        "model_version": model_server.model_version,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model."""
    if model_server.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_version": model_server.model_version,
        "window_size": model_server.window_size,
        "architecture": {
            "total_params": model_server.model.count_params(),
            "layers": len(model_server.model.layers),
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate prediction for USD/INR price direction.

    Args:
        request: Prediction request with feature data

    Returns:
        Prediction response with direction and confidence
    """
    try:
        # Validate input
        features = model_server.validate_input(request.features)

        # Generate prediction
        result = model_server.predict(features)

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        model_server.logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 30)
    print("API IS ONLINE")
    print(
        f"URL: http://{config.deployment['api_host']}:{config.deployment['api_port']}"
    )
    print("Routes registered: /, /health, /model-info, /predict")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.deployment["api_host"],
        port=config.deployment["api_port"],
        log_level="info",
    )
