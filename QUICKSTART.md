# Quick Start Guide

Get the FX Prediction System running in minutes.

## 5-Minute Setup

### 1. Install Dependencies (1 minute)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Run the Pipeline (3-4 minutes)

```bash
cd src
python main.py
```

This will:
- Download 18 years of multi-asset financial data
- Engineer features and select top predictors
- Train a CNN-LSTM model with 5-fold cross-validation
- Save trained model and experiment results

### 3. Deploy the API (30 seconds)

```bash
cd deployment
python api.py
```

### 4. Test the API (30 seconds)

In another terminal:
```bash
cd deployment
python test_api.py
```

Visit http://localhost:8000/docs for interactive API documentation.

## Docker Quick Start

If you have Docker installed:

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Test
curl http://localhost:8000/health
```

## Common First Steps

### Skip Data Download (if you already have data)

```bash
python main.py --skip-download --skip-preprocessing
```

### Enable Experiment Tracking

```bash
# Login to W&B first
wandb login

# Run with tracking
python main.py --use-wandb
```

### View Training Progress

In another terminal:
```bash
tensorboard --logdir ../logs/tensorboard
```

Open http://localhost:6006 in your browser.

## Typical Workflow

1. **First Run**: Complete pipeline
   ```bash
   python main.py
   ```

2. **Experiment**: Modify config, retrain
   ```bash
   # Edit config/config.yaml
   python main.py --skip-download --skip-preprocessing
   ```

3. **Deploy**: Start API server
   ```bash
   cd deployment
   python api.py
   ```

4. **Integrate**: Use API in your application
   ```python
   import requests
   response = requests.post(
       "http://localhost:8000/predict",
       json={"features": your_features}
   )
   ```

## Configuration Basics

Edit `config/config.yaml` to customize:

```yaml
model:
  window_size: 30        # Sequence length
  lstm_units: [128, 64]  # Network size
  learning_rate: 0.0005  # Training speed
  batch_size: 32         # Memory vs speed tradeoff

training:
  cv_splits: 5           # Cross-validation folds
```

## Expected Performance

- **Training Time**: ~5-10 minutes (CPU)
- **Directional Accuracy**: 55-58%
- **Model Size**: ~5 MB
- **Inference Time**: <100ms per prediction

## Troubleshooting

**"Module not found"**
→ Make sure venv is activated and dependencies installed

**"Data file not found"**
→ Run `python main.py --download-only` first

**"Model not loaded" in API**
→ Train a model first with `python main.py`

**API won't start**
→ Check port 8000 isn't already in use

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Review [CHANGELOG.md](CHANGELOG.md) for feature details
- Customize `config/config.yaml` for your needs
- Explore API at http://localhost:8000/docs

## Getting Help

Common issues and solutions:

1. **Out of memory**: Reduce `batch_size` in config
2. **Slow training**: Use fewer `cv_splits` or smaller `window_size`
3. **Poor accuracy**: Adjust `correlation_threshold` or add more features
4. **API errors**: Check logs in `logs/` directory

For data-related questions, check the data acquisition logs.
For model issues, review training logs and TensorBoard metrics.
