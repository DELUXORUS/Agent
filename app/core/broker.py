from faststream.rabbit import RabbitBroker
from app.config import settings

# Обернем брокер так, чтобы он сам мягко переподключался при старте
broker = RabbitBroker(settings.RABBITMQ_URL)