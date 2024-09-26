FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

RUN apt-get update && apt-get install -y \
    slurm-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN apt-get clean

CMD ["python3", "toxic-bert/main.py"]
