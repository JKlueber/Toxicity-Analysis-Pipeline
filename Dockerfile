FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

WORKDIR /app

COPY . /app
