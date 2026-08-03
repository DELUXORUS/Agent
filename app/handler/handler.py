# app/handler/main.py (твой handler)
import time
import asyncio
import logging
from faststream import FastStream
from langchain_core.messages import HumanMessage

from app.agent.graph import graph
from app.agent.state import AgentState
from app.core.broker import broker
from app.db.database import async_session_maker
from app.db.operations import Operations
from app.services.telegram import telegram_service

logger = logging.getLogger("uvicorn")
app = FastStream(broker)


async def generate_response(text: str, username: str, user_id: int) -> tuple[str, list]:
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
        found_movies = final_state.get("found_movies", [])
        logger.info(f"Шаг поиска выполнен за {time.perf_counter() - start_time:.2f} сек")

        if not response_text:
            return "К сожалению, не удалось сформировать ответ. Попробуй ещё раз!", []

        return response_text, found_movies

    except Exception as e:
        logger.error(f"Ошибка при выполнении LangGraph: {e}", exc_info=True)
        return "Произошла ошибка при обработке запроса. Попробуй позже!", []


@broker.subscriber("telegram_messages")
async def handle_telegram_messages(message: dict):
    if not message or not message.get("text"):
        return

    text = message["text"]
    user_id = message.get("user_id", 0)
    username = message.get("username") or f"id_{user_id}"
    chat_id = message.get("chat_id")

    if not chat_id:
        return

    reply_text, found_movies = await generate_response(text, username, user_id)
    keyboard = telegram_service.make_movies_keyboard(found_movies)

    try:
        await telegram_service.send_message(chat_id, reply_text, reply_markup=keyboard)
        logger.info(f"Ответ успешно отправлен в chat_id {chat_id}")
    except Exception as e:
        logger.error(f"Не удалось отправить ответ в chat_id {chat_id}: {e}")


@broker.subscriber("telegram_callbacks")
async def handle_telegram_callbacks(callback_data: dict):
    cb_id = callback_data.get("callback_query_id")
    cb_text = callback_data.get("callback_data", "")
    user_id = callback_data.get("user_id")
    chat_id = callback_data.get("chat_id")
    message_id = callback_data.get("message_id")

    if cb_text == "ignore":
        if cb_id:
            await telegram_service.answer_callback_query(cb_id, "Этот фильм уже отмечен!")
        return

    if not cb_text.startswith("watch:"):
        return

    try:
        _, movie_id_str, idx_str = cb_text.split(":")
        movie_id = int(movie_id_str)
    except ValueError:
        logger.warning(f"Некорректный формат callback_data: {cb_text}")
        return


    async with async_session_maker() as session:
        ops = Operations(session)
        await ops.add_movies_to_user_history(user_id=user_id, movie_ids=[movie_id])


    if cb_id:
        await telegram_service.answer_callback_query(
            cb_id, f"Фильм №{idx_str} добавлен в просмотренные!"
        )


    reply_markup = callback_data.get("reply_markup") or callback_data.get("message", {}).get("reply_markup", {})
    current_keyboard = reply_markup.get("inline_keyboard", [])

    if current_keyboard and chat_id and message_id:
        new_keyboard = []
        for row in current_keyboard:
            new_row = []
            for btn in row:
                if btn.get("callback_data") == cb_text:
                    new_row.append({"text": f"✅ №{idx_str}", "callback_data": "ignore"})
                else:
                    new_row.append(btn)
            new_keyboard.append(new_row)

        await telegram_service.edit_reply_markup(
            chat_id=chat_id,
            message_id=message_id,
            reply_markup={"inline_keyboard": new_keyboard}
        )


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