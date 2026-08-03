# app/services/telegram.py
import asyncio
import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import settings

logger = logging.getLogger("uvicorn")

class TelegramRateLimitError(Exception):
    def __init__(self, retry_after: int):
        self.retry_after = retry_after

class TelegramService:
    def __init__(self):
        self.base_url = f"https://api.telegram.org/bot{settings.BOT_TOKEN}"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, TelegramRateLimitError)),
        reraise=True,
    )
    async def send_message(self, chat_id: int, text: str, reply_markup: dict = None) -> None:
        url = f"{self.base_url}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
        if reply_markup:
            payload["reply_markup"] = reply_markup

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 429:
                retry_after = response.json().get("parameters", {}).get("retry_after", 1)
                await asyncio.sleep(retry_after)
                raise TelegramRateLimitError(retry_after)
            response.raise_for_status()

    async def answer_callback_query(self, callback_query_id: str, text: str) -> None:
        url = f"{self.base_url}/answerCallbackQuery"
        payload = {"callback_query_id": callback_query_id, "text": text, "show_alert": False}
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload)

    async def edit_reply_markup(self, chat_id: int, message_id: int, reply_markup: dict) -> None:
        url = f"{self.base_url}/editMessageReplyMarkup"
        payload = {"chat_id": chat_id, "message_id": message_id, "reply_markup": reply_markup}
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(url, json=payload)

    @staticmethod
    def make_movies_keyboard(movies: list) -> dict | None:
        if not movies:
            return None

        buttons = []
        for idx, movie in enumerate(movies, start=1):
            movie_id = movie.get("id") if isinstance(movie, dict) else getattr(movie, "id", None)

            if movie_id:
                buttons.append({
                    "text": f"👁 Смотрел №{idx}",
                    "callback_data": f"watch:{movie_id}:{idx}"
                })

        if not buttons:
            return None

        if len(buttons) <= 3:
            inline_keyboard = [buttons]
        else:
            inline_keyboard = [buttons[i:i + 3] for i in range(0, len(buttons), 3)]

        return {"inline_keyboard": inline_keyboard}

telegram_service = TelegramService()