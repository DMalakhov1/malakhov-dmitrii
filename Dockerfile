# Базовый образ Python
FROM python:3.13.5-slim

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Установка системных зависимостей для LightGBM и др. ML-библиотек
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копирование файла с зависимостями
COPY requirements.txt .

# Установка всех необходимых Python-пакетов
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всех файлов проекта в контейнер
COPY . .

# Команда запуска вашего решения
CMD ["python", "main.py"]

