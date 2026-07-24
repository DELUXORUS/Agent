import logging
from openai import AsyncOpenAI
from app.config import settings

logger = logging.getLogger("uvicorn")

client = AsyncOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

SYSTEM_PROMPT = """
Ты — вежливый и умный AI-ассистент в Telegram-боте.
Отвечай структурированно, понятным языком и по существу.
"""

async def request_to_llm(prompt: str) -> str:
    try:
        response = await client.chat.completions.create(
            model="google/gemma-4-26b-a4b-it:free",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "Не удалось сформировать ответ."

    except Exception as e:
        logger.error(f"Ошибка при запросе к OpenRouter LLM: {e}")
        return "Извини, произошла ошибка при генерации ответа. Попробуй позже."