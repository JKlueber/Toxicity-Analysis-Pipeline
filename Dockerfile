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

# Force numpy version to 1.25.2
RUN pip uninstall -y numpy && pip install numpy==1.25.2

WORKDIR /tmp

RUN apt-get clean

CMD ["python3", "toxic-bert/main.py"]
