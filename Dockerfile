# CUDA base image with dev tools + runtime + NVIDIA driver libs
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV RAY_memory=auto
ENV RAY_cpu=auto
ENV RAY_gpu=auto
ENV PYTHONPATH="/app/src"

# CUDA toolkit paths for Numba/cuDF
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python --version

# Install cuDF + RAPIDS ecosystem (with CUDA 12.4 compatibility)
RUN pip install --extra-index-url=https://pypi.nvidia.com \
    cudf-cu12 dask-cudf-cu12 --prefer-binary --no-cache-dir

RUN pip install --no-cache-dir numba==0.58.1

# Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir Flask gdown && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# App source
COPY ./src /app/src
COPY ./scripts /app/scripts

# Make scripts executable
RUN chmod +x /app/scripts/run_all.sh

# Entrypoint
ENTRYPOINT ["bash", "/app/scripts/run_all.sh"]