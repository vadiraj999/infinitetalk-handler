# ============================================================
# Dockerfile — InfiniteTalk Serverless (no ComfyUI)
# Base: CUDA 12.1 + Python 3.10
# Weights are NOT baked in — they live on the network volume
# ============================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# ── Clone InfiniteTalk repo ───────────────────────────────────
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /infinitetalk

WORKDIR /infinitetalk

# ── Python dependencies ───────────────────────────────────────
# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# xformers for attention optimization
RUN pip install --no-cache-dir \
    -U xformers==0.0.28 \
    --index-url https://download.pytorch.org/whl/cu121

# InfiniteTalk requirements
RUN pip install --no-cache-dir \
    ninja \
    psutil \
    packaging \
    wheel \
    misaki[en] \
    librosa \
    soundfile \
    huggingface_hub

RUN pip install --no-cache-dir -r /infinitetalk/requirements.txt

# flash-attn (compiled — takes a few minutes)
RUN pip install --no-cache-dir flash_attn==2.7.4.post1

# RunPod SDK
RUN pip install --no-cache-dir runpod

# ── Copy handler ──────────────────────────────────────────────
COPY handler.py /handler.py

# ── Environment defaults ──────────────────────────────────────
# WEIGHTS_DIR can be overridden in RunPod endpoint settings
ENV WEIGHTS_DIR=/workspace/weights

# ── Entry point ───────────────────────────────────────────────
CMD ["python", "/handler.py"]
