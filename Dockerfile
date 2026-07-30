FROM python:3.11-slim

# Отключаем создание .pyc файлов и включаем мгновенный вывод логов в консоль
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Копируем всю папку требований (чтобы app.txt мог подтянуть base.txt)
COPY requirements/ /app/requirements/

# 3. Устанавливаем ТОЛЬКО легкие зависимости для контейнера (CPU FastEmbed)
RUN pip install --no-cache-dir -r /app/requirements/docker.txt

# 4. Копируем исходный код проекта
COPY . .

# 5. Команда по умолчанию
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]