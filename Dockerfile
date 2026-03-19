FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set python alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Clone InfiniteTalk repo and patch multitalK.py
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /infinitetalk
RUN sed -i 's/from inspect import ArgSpec/from inspect import FullArgSpec as ArgSpec/' /infinitetalk/wan/multitalk.py

WORKDIR /infinitetalk

# Install PyTorch + xformers + requirements
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir -r /infinitetalk/requirements.txt
RUN pip install --no-cache-dir soundfile librosa "misaki[en]" runpod huggingface_hub transformers

# Copy your handler
COPY handler.py /handler.py

# Set cache and weights directories
ENV WEIGHTS_DIR=/workspace/weights
ENV HF_HOME=/workspace/weights/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/weights/hf_cache
ENV TRANSFORMERS_OFFLINE=1   # offline mode, use only preloaded cache

# Make sure weights and HF cache folders exist
RUN mkdir -p /workspace/weights/hf_cache

# -------------------------------
# COPY YOUR PRE-DOWNLOADED HF CACHE ZIP
# -------------------------------
# Make sure you have hf_cache.zip in the same folder as this Dockerfile
ADD hf_cache.zip /tmp/hf_cache.zip
RUN unzip /tmp/hf_cache.zip -d /workspace/weights/hf_cache \
    && rm /tmp/hf_cache.zip

# Lock cache permissions to prevent corruption
RUN chmod -R 555 /workspace/weights/hf_cache

# Run your handler
CMD ["python", "/handler.py"]
