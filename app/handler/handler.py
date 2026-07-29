import httpx
import asyncio
import logging
from faststream import FastStream
from langchain_core.messages import HumanMessage
from tenacity import (
    retry, stop_after_attempt,
    wait_exponential, retry_if_exception_type
)

from app.core.broker import broker
from app.config import settings
from app.agent.graph import graph
from app.agent.state import AgentState


logger = logging.getLogger("uvicorn")

app = FastStream(broker)

async def generate_response(text: str, username: str, user_id: int) -> str:
    logger.info(f"Запуск LangGraph агента для @{username} (id: {user_id}): '{text}'")

    # 1. Заполняем начальное состояние AgentState
    initial_state: AgentState = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "intent": None,
        "parsed_filter": None,
        "found_movies": [],
        "error_reason": None,
        "final_response": None,
    }

    try:
        final_state = await graph.ainvoke(initial_state)

        response_text = final_state.get("final_response")

        if not response_text:
            logger.warning("Граф завершился без заполненого final_response")
            return "К сожалению, не удалось сформировать ответ. Попробуй ещё раз!"

        logger.info(f"Ответ агента для @{username} успешно сгенерирован")
        return response_text

    except Exception as e:
        logger.error(f"Ошибка при выполнении LangGraph: {e}", exc_info=True)
        return "Произошла ошибка при обработке запроса. Попробуй позже!"


class TelegramRateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

# Декоратор автоматически повторит запрос с экспоненциальной задержкой,
# если Telegram вернет 429 или произойдет сетевая ошибка
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, TelegramRateLimitError))
)
async def send_telegram_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML"
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)

        # Обработка 429 Too Many Requests
        if response.status_code == 429:
            retry_after = response.json().get("parameters", {}).get("retry_after", 1)
            raise TelegramRateLimitError(retry_after)

        response.raise_for_status()

@broker.subscriber("telegram_messages")
async def handle_telegram_messages(message: dict):
    if not message:
        logger.info("Получено сообщение без текста (стикер/фото), пропуск.")
        return

    logger.info(
        f"Обработка сообщения от {message['username']}: '{message['text']}'"
    )

    reply_text = await generate_response(message["text"], message["username"], message["user_id"])
    success = await send_telegram_response(message["chat_id"], reply_text)

    if success:
        logger.info(f"Ответ успешно отправлен в chat_id {message['chat_id']}")
    else:
        logger.error(f"Ошибка при отправке ответа в chat_id {message['chat_id']}")



async def main():
    logger.info("Запуск FastStream Handler...")
    while True:
        try:
            await app.run()
            break
        except Exception as e:
            logger.warning(f"Ожидание RabbitMQ ({e}). Повтор через 3 секунды...")
            await asyncio.sleep(3)


if __name__ == "__main__":
    asyncio.run(main())