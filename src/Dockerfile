# Stage 1: Build stage
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime AS builder

# Install necessary build tools and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# Stage 2: Runtime stage
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from the build stage
COPY --from=builder /usr/local /usr/local

# Set working directory and copy application files
WORKDIR /app
COPY . /app

# Optional: Clean up
RUN apt-get clean

# Define entry point
CMD ["python3", "toxic-bert/main.py"]

