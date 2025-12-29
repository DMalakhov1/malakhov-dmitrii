# Базовый образ Python
FROM python:3.13.5-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["python", "main.py"]

