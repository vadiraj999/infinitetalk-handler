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

# Fix 2: Force tokenizers to load from local weights folder
RUN sed -i '1i import os' /infinitetalk/wan/modules/tokenizers.py
RUN sed -i 's|AutoTokenizer.from_pretrained(name,|AutoTokenizer.from_pretrained(os.path.join(os.environ.get("WEIGHTS_DIR","/runpod-volume/weights"), name), local_files_only=True,|g' /infinitetalk/wan/modules/tokenizers.py

# Fix 3: Rename google/umt5-xxl to google-umt5-xxl to avoid path separator issues
RUN grep -rl 'google/umt5-xxl' /infinitetalk | xargs sed -i 's|google/umt5-xxl|google-umt5-xxl|g'

# Fix 4: Use memory-mapped safe_open instead of load_file for large NFS files
RUN sed -i 's/from safetensors.torch import load_file/from safetensors.torch import load_file\nfrom safetensors import safe_open/' /infinitetalk/wan/multitalk.py
RUN sed -i 's/                    sd = load_file(weight_file)/                    with safe_open(weight_file, framework="pt", device="cpu") as _f:\n                        sd = {k: _f.get_tensor(k) for k in _f.keys()}/' /infinitetalk/wan/multitalk.py

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
    huggingface_hub \
    "transformers==4.49.0" \
    "safetensors==0.7.0" \
    protobuf

COPY handler.py /handler.py

ENV WEIGHTS_DIR=/runpod-volume/weights
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "/handler.py"]
