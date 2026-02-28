FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Создаем storage директорию
RUN mkdir -p /app/storage

ENV STORAGE_DIR=/app/storage
ENV WORKER_URL=http://worker:8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]