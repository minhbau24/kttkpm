# Use a lightweight python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU version first to keep the image size small
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file and install python packages
COPY web/requirements.txt /app/web/requirements.txt
RUN pip install --no-cache-dir -r /app/web/requirements.txt

# Pre-download and cache the sentence-transformers model during build
# This ensures that the model is already present in the Docker image and not downloaded at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy the project files
COPY data /app/data
COPY trainmodel /app/trainmodel
COPY web /app/web

# Set working directory to the web app
WORKDIR /app/web

# Expose port 8000
EXPOSE 8000

# Start Gunicorn server via the python entrypoint
CMD ["python", "entrypoint.py"]
