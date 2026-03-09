# Dockerfile (worker-service)
FROM python:3.10-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN mkdir -p /app/models /app/storage

# Финальный слой
FROM base

COPY app/ ./app/
COPY core/ ./core/

ENV MODELS_DIR=/app/models
ENV STORAGE_DIR=/app/storage
ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8041"]