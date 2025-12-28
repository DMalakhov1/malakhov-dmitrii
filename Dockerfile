# Базовый образ Python
FROM python:3.12-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка системных зависимостей для LightGBM и обновление pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]
