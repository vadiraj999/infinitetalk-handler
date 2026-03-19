FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /infinitetalk

RUN sed -i 's/from inspect import ArgSpec/from inspect import FullArgSpec as ArgSpec/' /infinitetalk/wan/multitalk.py

WORKDIR /infinitetalk

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r /infinitetalk/requirements.txt

RUN pip install --no-cache-dir \
    soundfile \
    librosa \
    "misaki[en]" \
    runpod \
    huggingface_hub

COPY handler.py /handler.py

ENV WEIGHTS_DIR=/runpod-volume/weights

CMD ["python", "/handler.py"]
