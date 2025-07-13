FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# NVIDIA runtime environment
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.0"

# Layer 1: System dependencies (משתנה רק לעתים רחוקות)
# כל החבילות של המערכת ופייטון בסיסי
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    libaio-dev \
    libopenmpi-dev \
    openmpi-bin \
    openmpi-common \
    libopenmpi3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Layer 2: Python dependencies (משתנה רק כשמעדכנים requirements)
# העתק requirements ותתקין חבילות פייטון
COPY requirements.txt /opt/ml/code/requirements.txt

# Install MPI4PY first (requires OpenMPI to be installed)
RUN pip install --no-cache-dir mpi4py>=3.1.0

# Install other requirements
RUN pip install --no-cache-dir -r /opt/ml/code/requirements.txt

# Install DeepSpeed with CUDA support (after other packages)
# Use --global-option to ensure proper compilation
RUN DS_BUILD_OPS=1 pip install --no-cache-dir deepspeed>=0.12.0 --global-option="build_ext" --global-option="-j8"

# Verify installations
RUN python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
RUN python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
RUN python -c "import mpi4py; print(f'MPI4PY version: {mpi4py.__version__}')"

# Layer 3: Configuration files (משתנה רק כשמעדכנים config)
# העתק קבצי קונפיגורציה שמשתנים פחות
COPY configs/ /opt/ml/code/configs/

# Layer 4: Training scripts (זה משתנה הכי הרבה - לכן בסוף!)
# רק עכשיו נעתיק את קבצי הקוד שמשתנים הכי הרבה
COPY scripts/ /opt/ml/code/

# Set working directory
WORKDIR /opt/ml/code

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# DeepSpeed specific environment variables
ENV DS_BUILD_OPS=1
ENV DS_BUILD_AIO=1
ENV DS_BUILD_FUSED_ADAM=1
ENV DS_BUILD_FUSED_LAMB=1
ENV DS_BUILD_TRANSFORMER=1
ENV DS_BUILD_STOCHASTIC_TRANSFORMER=1

# Entry point
ENTRYPOINT ["python", "train.py"]