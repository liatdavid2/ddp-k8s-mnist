FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Basic OS deps (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.txt /app/requirements.txt

# Install CPU wheels for torch/torchvision
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/requirements.txt --index-url https://download.pytorch.org/whl/cpu

COPY src /app/src

# Default command can be overridden by k8s args
CMD ["python", "-u", "src/train.py"]