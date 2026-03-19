FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /infinitetalk

WORKDIR /infinitetalk

RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    -U xformers==0.0.28 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    ninja \
    psutil \
    packaging \
    wheel \
    "misaki[en]" \
    librosa \
    soundfile \
    huggingface_hub

RUN pip install --no-cache-dir -r /infinitetalk/requirements.txt

RUN pip uninstall -y xfuser || true

RUN pip install --no-cache-dir diffusers==0.30.3 transformers accelerate

RUN pip install --no-cache-dir flash_attn==2.7.4.post1

RUN pip install --no-cache-dir runpod

COPY handler.py /handler.py

ENV WEIGHTS_DIR=/workspace/weights

CMD ["python", "/handler.py"]
