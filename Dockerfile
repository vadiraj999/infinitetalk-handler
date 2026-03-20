FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.11 and dependencies
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

# Clone InfiniteTalk repo
RUN git clone https://github.com/MeiGen-AI/InfiniteTalk.git /infinitetalk

# ===== FIXES =====
# Fix 1: ArgSpec removed in Python 3.11
RUN sed -i 's/from inspect import ArgSpec/from inspect import FullArgSpec as ArgSpec/' /infinitetalk/wan/multitalk.py

# Fix 2: Force all tokenizers to load from local weights dynamically
RUN sed -i 's|AutoTokenizer.from_pretrained(name,|AutoTokenizer.from_pretrained("/runpod-volume/weights/" + name, local_files_only=True,|g' /infinitetalk/wan/modules/tokenizers.py

# Fix 3: Replace UMT5 path everywhere
RUN grep -rl 'google/umt5-xxl' /infinitetalk | xargs sed -i 's|google/umt5-xxl|/runpod-volume/weights/google-umt5-xxl|g'

# Fix 4: Replace XLM-R path everywhere
RUN grep -rl 'xlm-roberta-large' /infinitetalk | xargs sed -i 's|xlm-roberta-large|/runpod-volume/weights/xlm-roberta-large|g'

# Set working directory
WORKDIR /infinitetalk

# Install PyTorch and xformers
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    xformers==0.0.28.post3 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN pip install --no-cache-dir -r /infinitetalk/requirements.txt
RUN pip install --no-cache-dir \
    soundfile \
    librosa \
    "misaki[en]" \
    runpod \
    huggingface_hub \
    "transformers==4.49.0" \
    protobuf

# Copy handler script
COPY handler.py /handler.py

# Set environment variables for local weights and cache
ENV WEIGHTS_DIR=/runpod-volume/weights
ENV HF_HOME=/runpod-volume/hf_cache
ENV TRANSFORMERS_CACHE=/runpod-volume/hf_cache
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Run the handler
CMD ["python", "/handler.py"]
