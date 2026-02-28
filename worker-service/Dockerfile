FROM python:3.10-slim

WORKDIR /app

# Системные зависимости для OpenCV и PaddleOCR (исправлено для Debian trixie)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .

# Устанавливаем Python-пакеты
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY app/ ./app/
COPY core/ ./core/

# Создаем директории для моделей и storage
RUN mkdir -p /app/models /app/storage

# Переменные окружения
ENV MODELS_DIR=/app/models
ENV STORAGE_DIR=/app/storage
ENV PYTHONPATH=/app

# Запускаем
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]