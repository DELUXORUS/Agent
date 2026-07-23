import httpx
import asyncio
import logging
from app.config import settings
from faststream import FastStream
from app.core.broker import broker


logger = logging.getLogger("uvicorn")

app = FastStream(broker)


async def generate_response(text: str, username: str) -> str:
    return f"🤖 Привет, {username}! Я получил твое сообщение: '{text}'"


async def send_telegram_response(chat_id: int, text: str):
    url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        return response.status_code == 200

@broker.subscriber("telegram_messages")
async def handle_telegram_messages(message: dict):
    if not message:
        logger.info("Получено сообщение без текста (стикер/фото), пропуск.")
        return

    logger.info(
        f"[handler.handle_telegram_messages] Обработка сообщения от {message['username']}: '{message['text']}'"
    )

    reply_text = await generate_response(message["text"], message["username"])
    success = await send_telegram_response(message["chat_id"], reply_text)

    if success:
        logger.info(f"[handler.handle_telegram_messages] Ответ успешно отправлен в chat_id {message['chat_id']}")
    else:
        logger.error(f"[handler.handle_telegram_messages] Ошибка при отправке ответа в chat_id {message['chat_id']}")



async def main():
    logger.info("Запуск FastStream Handler...")
    while True:
        try:
            await app.run()
            break
        except Exception as e:
            logger.warning(f"⏳ Ожидание RabbitMQ ({e}). Повтор через 3 секунды...")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())