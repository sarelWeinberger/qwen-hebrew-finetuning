FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Layer 2: Python dependencies (SIMPLIFIED - no DeepSpeed)
COPY requirements.txt /opt/ml/code/requirements.txt

# Install requirements (much simpler now)
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Verify core installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
RUN python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"

# Layer 3: Training script - FIXED PATH
COPY scripts/train.py /opt/ml/code/train.py

# Set working directory
WORKDIR /opt/ml/code

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set CUDA environment variables for optimization
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# PyTorch optimizations
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Entry point
ENTRYPOINT ["python", "train.py"]