# 1. Use a slim Python image
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy ONLY requirements first (Crucial for build speed)
COPY requirements.txt .


RUN pip install uv
# 5. Install Python dependencies


RUN uv pip install --no-cache-dir --system -r requirements.txt



# 6. Copy the rest of the project
COPY . .

# 7. Create necessary directories
RUN mkdir -p data models logs/tensorboard

# 8. CRITICAL: Set PYTHONPATH so 'src' is findable
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 9. Expose API port
EXPOSE 8001

# 10. Run as a module to fix the sibling import issue
CMD ["python", "-m", "deployment.api"]