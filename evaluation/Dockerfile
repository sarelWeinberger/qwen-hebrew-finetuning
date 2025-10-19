FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages - include ALL lighteval dependencies
RUN pip install --no-cache-dir \
    huggingface_hub \
    transformers \
    accelerate \
    lighteval[extended_tasks] \
    vllm \
    boto3 \
    awscli \
    emoji \
    nltk \
    python-Levenshtein

# Download NLTK data (lighteval needs it)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy the benchmark script
COPY run_benchmark_vllm_sm.sh /opt/ml/code/train
RUN chmod +x /opt/ml/code/train

WORKDIR /opt/ml/code
ENTRYPOINT ["bash", "/opt/ml/code/train"]