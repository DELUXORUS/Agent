FROM python:3.11-slim

# 2. Отключаем кэширование и буферизацию логов Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# 6. Команда по умолчанию при старте
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]