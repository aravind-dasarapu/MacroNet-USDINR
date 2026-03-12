# Deployment Guide

This guide covers different deployment options for the FX Prediction System.

## Local Development Deployment

### Prerequisites
- Python 3.8+
- Virtual environment
- Required dependencies installed

### Steps

1. Train the model first:
```bash
cd src
python main.py
```

2. Start the API server:
```bash
cd deployment
python api.py
```

3. Test the API:
```bash
python test_api.py
```

The API will be available at `http://localhost:8000`

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed (optional but recommended)

### Using Docker Compose (Recommended)

1. Build and start the container:
```bash
docker-compose up --build -d
```

2. Check logs:
```bash
docker-compose logs -f
```

3. Stop the container:
```bash
docker-compose down
```

### Using Docker Directly

1. Build the image:
```bash
docker build -t fx-prediction:latest .
```

2. Run the container:
```bash
docker run -d \
  --name fx-prediction-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  fx-prediction:latest
```

3. View logs:
```bash
docker logs -f fx-prediction-api
```

4. Stop the container:
```bash
docker stop fx-prediction-api
docker rm fx-prediction-api
```

## Cloud Deployment

### AWS EC2

1. Launch an EC2 instance (t2.medium or larger recommended)
2. Install Docker on the instance
3. Copy your project files to the instance
4. Run using Docker Compose
5. Configure security groups to allow port 8000

### Google Cloud Run

1. Build the Docker image:
```bash
docker build -t gcr.io/YOUR_PROJECT/fx-prediction:latest .
```

2. Push to Google Container Registry:
```bash
docker push gcr.io/YOUR_PROJECT/fx-prediction:latest
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy fx-prediction \
  --image gcr.io/YOUR_PROJECT/fx-prediction:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure Container Instances

1. Create a container registry
2. Build and push image
3. Deploy using Azure CLI:
```bash
az container create \
  --resource-group YOUR_RESOURCE_GROUP \
  --name fx-prediction \
  --image YOUR_REGISTRY/fx-prediction:latest \
  --ports 8000 \
  --cpu 2 \
  --memory 4
```

## Production Considerations

### Performance
- Use GPU-enabled instances for faster inference
- Consider model quantization for reduced latency
- Implement caching for frequently requested predictions
- Use load balancing for high traffic scenarios

### Security
- Enable HTTPS/TLS
- Implement API key authentication
- Use environment variables for sensitive configuration
- Enable rate limiting
- Validate all inputs thoroughly

### Monitoring
- Set up application performance monitoring (APM)
- Configure logging aggregation
- Set up alerts for API errors
- Monitor model performance drift

### Scaling
- Use container orchestration (Kubernetes) for auto-scaling
- Implement horizontal pod autoscaling
- Use managed services for reduced operational overhead

## Environment Variables

Key environment variables for deployment:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_DIR=/app/models
DATA_DIR=/app/data

# Logging
LOG_LEVEL=INFO
LOG_DIR=/app/logs

# TensorFlow Configuration
TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0  # If using GPU
```

## Health Checks

The API provides a health check endpoint at `/health`:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "fx_experiment_20240311_103000_final",
  "timestamp": "2024-03-11T10:30:00"
}
```

## Troubleshooting

### Container won't start
- Check that model files are in the correct directory
- Verify all volumes are properly mounted
- Check container logs: `docker logs fx-prediction-api`

### API returning 503
- Model may not be loaded
- Check that model files exist in the models directory
- Verify model file permissions

### High memory usage
- Reduce batch size in configuration
- Use model quantization
- Implement request queuing

### Slow predictions
- Use GPU-accelerated instance
- Optimize model architecture
- Implement model caching
- Consider using TensorFlow Lite for inference

## Backup and Recovery

### Model Backup
```bash
# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Restore from backup
tar -xzf models_backup_YYYYMMDD.tar.gz
```

### Data Backup
```bash
# Backup data directory
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

## Updating the Model

1. Train a new model version:
```bash
cd src
python main.py
```

2. The API will automatically load the newest model on restart:
```bash
docker-compose restart
```

Or update without downtime using rolling deployment in Kubernetes.