from faststream.rabbit import RabbitBroker
from app.config import settings

broker = RabbitBroker(settings.RABBITMQ_URL)

