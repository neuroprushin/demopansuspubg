# Используем базовый образ с Python
FROM python:3.9-slim

# Устанавливаем системные зависимости для OpenCV и MediaPipe
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY . .

# Указываем порт
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "app.py"]