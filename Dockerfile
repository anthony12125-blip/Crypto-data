FROM runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04

WORKDIR /workspace

# Install dependencies
RUN pip install --no-cache-dir mlx mlx-lm transformers

# Copy all files
COPY . /workspace/

# Set environment
ENV PYTHONUNBUFFERED=1

# Run training for all 3 agents
CMD ["python3", "train_all_agents.py"]
