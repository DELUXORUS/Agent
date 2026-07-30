import time
import httpx
import logging
import asyncio
from faststream import FastStream
from langchain_core.messages import HumanMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.agent.graph import graph
from app.agent.state import AgentState
from app.config import settings
from app.core.broker import broker

logger = logging.getLogger("uvicorn")

app = FastStream(broker)


async def generate_response(text: str, username: str, user_id: int) -> str:
    logger.info(f"Запуск LangGraph агента для @{username} (id: {user_id}): '{text}'")

    initial_state: AgentState = {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "intent": None,
        "parsed_filter": None,
        "found_movies": [],
        "error_reason": None,
        "final_response": None,
    }

    start_time = time.perf_counter()

    try:
        final_state = await graph.ainvoke(initial_state)
        response_text = final_state.get("final_response")
        logger.info(f"Шаг поиска выполнен за {time.perf_counter() - start_time:.2f} сек")

        if not response_text:
            logger.warning("Граф завершился без заполненного final_response")
            return "К сожалению, не удалось сформировать ответ. Попробуй ещё раз!"

        logger.info(f"Ответ агента для @{username} успешно сгенерирован")
        return response_text

    except Exception as e:
        logger.error(f"Ошибка при выполнении LangGraph: {e}", exc_info=True)
        return "Произошла ошибка при обработке запроса. Попробуй позже!"


class TelegramRateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((httpx.HTTPError, TelegramRateLimitError)),
    reraise=True,
)
async def send_telegram_message(chat_id: int, text: str) -> None:
    url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=payload)

        if response.status_code == 429:
            retry_after = response.json().get("parameters", {}).get("retry_after", 1)
            logger.warning(f"Telegram Rate Limit (429). Ожидание {retry_after} сек...")
            await asyncio.sleep(retry_after)
            raise TelegramRateLimitError(retry_after)

        response.raise_for_status()


@broker.subscriber("telegram_messages")
async def handle_telegram_messages(message: dict):
    if not message or not message.get("text"):
        logger.info("Получено сообщение без текста (стикер/фото/пусто), пропуск.")
        return

    text = message["text"]
    user_id = message.get("user_id", 0)
    username = message.get("username") or f"id_{user_id}"
    chat_id = message.get("chat_id")

    if not chat_id:
        logger.error("В сообщении отсутствует chat_id!")
        return

    logger.info(f"Обработка сообщения от @{username}: '{text}'")

    reply_text = await generate_response(text, username, user_id)

    try:
        # 🐛 ИСПРАВЛЕНО: Вызываем верное имя функции send_telegram_message
        await send_telegram_message(chat_id, reply_text)
        logger.info(f"Ответ успешно отправлен в chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Не удалось отправить ответ в chat_id {chat_id}: {e}")


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