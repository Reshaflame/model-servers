# Base image with CUDA 12.4 runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV RAY_memory=auto
ENV RAY_cpu=auto
ENV RAY_gpu=auto
ENV PYTHONPATH="/app/src"

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

# Set work directory inside the container
WORKDIR /app

# Copy only the requirements file first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir Flask \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

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
