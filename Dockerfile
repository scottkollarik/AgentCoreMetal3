# Use Python 3.10 slim image with platform specification
FROM --platform=linux/arm64/v8 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set base environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Platform-specific configurations
ARG TARGETPLATFORM
ARG USE_GPU=false

RUN if [ "$USE_GPU" = "true" ]; then \
        if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
            echo "Configuring for Apple Silicon (M4) with Metal 3" && \
            # Metal 3 specific optimizations \
            export PYTORCH_ENABLE_MPS_FALLBACK=1 && \
            export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 && \
            export PYTORCH_MPS_ALLOCATOR_POLICY=default && \
            export PYTORCH_MPS_DEVICE=0 && \
            export PYTORCH_MPS_CACHE_SIZE=0 && \
            # Performance tuning \
            export PYTORCH_MPS_USE_METAL_SHADER_CACHE=1 && \
            export PYTORCH_MPS_USE_METAL_COMPILE_OPTIONS=1; \
        elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
            echo "Configuring for Linux x86_64 with CUDA" && \
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512; \
        elif [ "$TARGETPLATFORM" = "windows/amd64" ]; then \
            echo "Configuring for Windows x86_64 with CUDA" && \
            export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512; \
        fi \
    else \
        echo "Configuring for CPU-only setup" && \
        export PYTORCH_CUDA_VISIBLE_DEVICES="" && \
        export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; \
    fi

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with platform-specific considerations
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p memory/chroma_db

# Expose port for API
EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 