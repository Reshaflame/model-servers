# Base image with CUDA 12.4 runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV RAY_memory=auto
ENV RAY_cpu=auto
ENV RAY_gpu=auto

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    nvidia-cuda-toolkit \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*


# Set python3.10 as default
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python --version

# Set work directory inside the container
WORKDIR /app

# Copy only the requirements file first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir nvidia-pyindex && \
    pip install --no-cache-dir nvidia-pytorch==2.5.0+nvidia_cuda12.4_cudnn9.1

# Copy your source code into the container
COPY ./src ./src

# Install gdown for Google Drive automation
RUN pip install --no-cache-dir gdown

# Copy your download script into the image
WORKDIR /app/scripts
COPY ./scripts/download_datasets.sh .
COPY ./scripts/run_all.sh .
RUN chmod +x run_all.sh
WORKDIR /app

# Define the entrypoint (you can modify based on your typical workflow)
ENTRYPOINT ["bash", "/app/scripts/run_all.sh"]
