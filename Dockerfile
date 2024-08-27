FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    make \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /app

COPY . /app
