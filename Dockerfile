FROM runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for Hugging Face + PEFT training
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.37.2 \
    datasets==2.16.1 \
    peft==0.8.2 \
    bitsandbytes==0.42.0 \
    accelerate==0.26.1 \
    scipy \
    sentencepiece \
    protobuf

# Copy all files
COPY . /workspace/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/workspace/.cache/huggingface

# Run data preparation first, then training
CMD ["bash", "-c", "python3 data_prep.py && python3 train_all_agents.py"]
