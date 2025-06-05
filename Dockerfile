# Use multi-stage build for better layer management
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Platform-specific configurations
ARG TARGETPLATFORM
ARG USE_GPU=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    graphviz-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Platform-specific optimizations
RUN if [ "$USE_GPU" = "false" ]; then \
        if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
            echo "Configuring for Apple Silicon (M4) CPU-only" && \
            export PYTORCH_ENABLE_MPS_FALLBACK=0 && \
            export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0; \
        elif [ "$TARGETPLATFORM" = "linux/amd64" ]; then \
            echo "Configuring for Linux x86_64 CPU-only" && \
            export PYTORCH_CUDA_VISIBLE_DEVICES="" && \
            export PYTORCH_CPU_THREADS=$(nproc); \
        elif [ "$TARGETPLATFORM" = "windows/amd64" ]; then \
            echo "Configuring for Windows x86_64 CPU-only" && \
            export PYTORCH_CUDA_VISIBLE_DEVICES="" && \
            export PYTORCH_CPU_THREADS=$(nproc); \
        fi \
    fi

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install wheel && \
    if [ "$USE_GPU" = "false" ]; then \
        pip install --no-cache-dir -r requirements.txt --no-deps && \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Final stage
FROM python:3.11-slim

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "-m", "app.main"] 