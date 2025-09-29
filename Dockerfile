FROM python:3.11-bookworm

RUN apt update && apt install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install realesrgan
COPY degradations.py /usr/local/lib/python3.11/site-packages/basicsr/data/degradations.py

RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
WORKDIR /app
