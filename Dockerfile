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

# Fix 1: ArgSpec removed in Python 3.11
RUN sed -i 's/from inspect import ArgSpec/from inspect import FullArgSpec as ArgSpec/' /infinitetalk/wan/multitalk.py

# Fix 2: Patch tokenizer to use local cache from env var
RUN sed -i 's/self\.tokenizer = AutoTokenizer\.from_pretrained(name,/self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=os.environ.get("HF_HOME"),/' /infinitetalk/wan/modules/tokenizers.py
RUN sed -i '1s/^/import os\n/' /infinitetalk/wan/modules/tokenizers.py

WORKDIR /infinitetalk

RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
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
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HF_HUB_OFFLINE=1

CMD ["python", "/handler.py"]
