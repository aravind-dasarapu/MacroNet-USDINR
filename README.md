# FX Prediction System: USD/INR Deep Learning Pipeline

A production-ready machine learning framework for forecasting USD/INR foreign exchange price direction using multi-asset macroeconomic drivers and deep learning.

## Overview

This system integrates global macroeconomic indicators including Brent Crude Oil, US Dollar Index (DXY), Nifty 50, and other assets to capture inter-market correlations for FX prediction. The architecture combines CNN and LSTM layers to extract both local temporal patterns and long-term sequential dependencies.

## Key Features

### Model Architecture
- Hybrid CNN-LSTM neural network for time series forecasting
- Custom directional loss function optimized for trading signals
- Temporal feature engineering with proper lag handling to prevent data leakage
- Time series cross-validation for robust performance estimation

### Data Processing
- Automated multi-asset data ingestion via Yahoo Finance API
- Stationary feature engineering using log-returns and normalized indicators
- Correlation-based feature selection
- Missing data handling with forward-fill and backward-fill strategies

### Experiment Tracking
- TensorBoard integration for real-time training monitoring
- Weights & Biases (W&B) support for comprehensive experiment tracking
- Automated model versioning and checkpointing
- Detailed logging and performance metrics

### Deployment
- FastAPI REST API for model inference
- Input validation and error handling
- Model versioning and health check endpoints
- Docker-ready architecture

## Project Structure

```
fx_prediction_enhanced/
├── config/
│   └── config.yaml              # Centralized configuration
├── src/
│   ├── config_loader.py         # Configuration management
│   ├── logger.py                # Logging utilities
│   ├── data_download.py         # Data acquisition module
│   ├── data_preprocessing.py    # Feature engineering
│   ├── model_training.py        # Model training with monitoring
│   └── main.py                  # Pipeline orchestrator
├── deployment/
│   └── api.py                   # FastAPI deployment server
├── data/                        # Data storage (generated)
├── models/                      # Trained models (generated)
├── logs/                        # Execution logs (generated)
└── requirements.txt             # Python dependencies
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download the project:
```bash
cd fx_prediction_enhanced
```

2. Create and activate virtual environment:
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline

Run the entire pipeline (download, preprocess, train):
```bash
cd src
python main.py
```

### With Options

Enable W&B experiment tracking:
```bash
python main.py --use-wandb
```

Skip data download if data exists:
```bash
python main.py --skip-download
```

Skip preprocessing if processed data exists:
```bash
python main.py --skip-preprocessing
```

### Individual Steps

Download data only:
```bash
python main.py --download-only
```

Preprocess data only:
```bash
python main.py --preprocess-only
```

Train model directly:
```bash
python model_training.py
```

### Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir ../logs/tensorboard
```

### Deployment

Start the API server:
```bash
cd deployment
python api.py
```

API will be available at `http://localhost:8000`

Test the API:
```bash
curl http://localhost:8000/health
```

API Documentation: `http://localhost:8000/docs`

## Configuration

All hyperparameters and settings are centralized in `config/config.yaml`:

- **Data**: Tickers, download period, file paths
- **Preprocessing**: Feature selection thresholds, technical indicators
- **Model**: Architecture parameters, training hyperparameters
- **Monitoring**: TensorBoard and W&B settings
- **Deployment**: API server configuration

Modify the configuration file to customize the pipeline without changing code.

## Model Performance

The model is optimized for directional accuracy rather than precise price prediction, as direction is more actionable for trading strategies.

Typical performance metrics:
- Directional Accuracy: 55-58% (across cross-validation folds)
- Validation MAE: 0.10-0.12
- Architecture: CNN-LSTM hybrid with batch normalization

Performance varies based on market conditions and the specific time period evaluated.

## Technical Details

### Data Integrity
The pipeline ensures temporal integrity through:
- T-1 feature lagging to prevent look-ahead bias
- T+1 target lead-shifting for proper prediction setup
- Time series cross-validation with no data leakage between folds

### Feature Engineering
- Log returns for stationarity
- Rolling volatility (5-day window)
- Distance from moving average
- Exponential weighted smoothing of targets

### Loss Function
Custom directional log-cosh loss that:
- Penalizes incorrect direction predictions more heavily
- Combines robustness of log-cosh with directional focus
- Optimizes for trading signal quality

## API Reference

### Endpoints

**GET /** - API information and available endpoints

**GET /health** - Health check and model status

**GET /model-info** - Model architecture and version details

**POST /predict** - Generate prediction
```json
{
  "features": [[0.1, 0.2, ...], [0.15, 0.25, ...], ...],
  "window_size": 30
}
```

Response:
```json
{
  "prediction": 0.045,
  "direction": "UP",
  "confidence": 0.45,
  "timestamp": "2024-03-11T10:30:00",
  "model_version": "fx_experiment_20240311_103000_final"
}
```

## Development

### Code Quality
- Comprehensive logging throughout the pipeline
- Type hints for better IDE support
- Modular design with clear separation of concerns
- Configuration-driven architecture

### Testing
Run unit tests (when available):
```bash
pytest tests/
```

### Extending the System

To add new features:
1. Update ticker configuration in `config/config.yaml`
2. Modify feature engineering in `data_download.py`
3. Adjust model architecture in `model_training.py`

## Troubleshooting

**Issue**: Module not found errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify virtual environment is activated

**Issue**: Data download failures
- Check internet connection
- Verify ticker symbols are valid
- Some tickers may have limited historical data

**Issue**: Model not loading in deployment
- Ensure model training completed successfully
- Check that model files exist in the `models/` directory
- Verify file permissions

**Issue**: W&B authentication errors
- Login to W&B: `wandb login`
- Or disable W&B in `config/config.yaml`

## Performance Optimization

For faster training:
- Reduce `cv_splits` in configuration
- Decrease `window_size` for less sequential context
- Use GPU-enabled TensorFlow installation
- Adjust `batch_size` based on available memory

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

Data provided by Yahoo Finance via the yfinance library.