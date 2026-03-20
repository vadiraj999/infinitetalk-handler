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

# Fix 2: Force tokenizer to load from local weights
RUN sed -i 's|AutoTokenizer.from_pretrained(name,|AutoTokenizer.from_pretrained("/runpod-volume/weights/xlm-roberta-large", local_files_only=True,|g' /infinitetalk/wan/modules/tokenizers.py

# Fix 3: Force UMT5 model to load locally
RUN grep -rl 'google/umt5-xxl' /infinitetalk | xargs sed -i 's|"google/umt5-xxl"|"/runpod-volume/weights/google-umt5-xxl"|g'

# Fix 4: Replace all xlm-roberta references
RUN grep -rl 'xlm-roberta-large' /infinitetalk | xargs sed -i 's|"xlm-roberta-large"|"/runpod-volume/weights/xlm-roberta-large"|g'

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

RUN pip install --no-cache-dir "transformers==4.49.0"

COPY handler.py /handler.py

ENV WEIGHTS_DIR=/runpod-volume/weights
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "/handler.py"]
