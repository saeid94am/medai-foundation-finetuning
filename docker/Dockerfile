FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

LABEL maintainer="Saeid Amini <saeed.amiinii@gmail.com>"
LABEL description="MedSAM LoRA — interactive breast lesion segmentation on BUSI"

# System deps required by OpenCV and general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency specs before source code to maximise layer cache reuse.
# A change to src/ will not invalidate the pip install layer.
COPY pyproject.toml README.md ./

# Install all dependencies including the local package in editable mode.
# torch/torchvision are already in the base image; pip will skip them.
RUN pip install --no-cache-dir -e ".[dev]"

# Copy the rest of the project
COPY configs/   configs/
COPY scripts/   scripts/
COPY src/       src/
COPY tests/     tests/

# Make scripts executable
RUN chmod +x scripts/*.sh

# Deterministic Python output — important for reproducible logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Weights & Biases offline mode by default inside container.
# Override with -e WANDB_MODE=online when running.
ENV WANDB_MODE=offline

# MedSAM weights path — mount or copy at runtime; do NOT bake into image.
# Expected: /workspace/checkpoints/medsam_vit_b.pth
ENV MEDSAM_CHECKPOINT=/workspace/checkpoints/medsam_vit_b.pth

CMD ["bash", "scripts/run_experiments.sh"]
