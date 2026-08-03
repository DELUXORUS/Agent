import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.state import AgentState, MovieFilter, UnifiedParseResult
from app.agent.tools import (
    fetch_guessed_movie_hybrid,
    fetch_recommended_movies,
    generate_query_embedding,
)
from app.schemas import MovieDTO
from app.agent.load_prompts import prompts


logger = logging.getLogger("uvicorn")


async def unified_parser_node(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    logger.info("Единый парсинг интента и фильтров (1 вызов LLM)...")

    raw_messages = state.get("messages") or []

    structured_llm = llm.with_structured_output(UnifiedParseResult)
    messages = [SystemMessage(content=prompts["SYSTEM_UNIFIED_PARSER_PROMPT"])] + list(raw_messages)

    try:
        parsed: UnifiedParseResult = await structured_llm.ainvoke(messages)
        logger.info(f"Интент: {parsed.intent} | Filter: {parsed.filter}")

        return {
            "intent": parsed.intent,
            "parsed_filter": parsed.filter,
        }
    except Exception as e:
        logger.error(f"Ошибка в unified_parser_node: {e}")
        return {
            "intent": "general_chat",
            "parsed_filter": MovieFilter(),
        }


async def search_for_recommended_node(state: AgentState) -> dict[str, Any]:
    filters: MovieFilter = state["parsed_filter"]

    query_vector = None
    if filters.is_semantic_search_needed and filters.query_text:
        logger.info(f"Векторизация очищенного текста: '{filters.query_text}'")
        query_vector = await generate_query_embedding(filters.query_text)
    else:
        logger.info("Поиск по точным SQL-фильтрам (без вектора)")

    found_movies: list[MovieDTO] = await fetch_recommended_movies(
        user_id=state["user_id"],
        movie_filter=filters,
        query_vector=query_vector,
        limit=3,
    )

    logger.info(f"В состояние занесено {len(found_movies)} фильмов.")

    return {
        "parsed_filter": filters,
        "found_movies": found_movies,
    }


async def search_for_guess_node(state: AgentState) -> dict[str, Any]:
    filters: MovieFilter = state["parsed_filter"]

    if (
            not filters.is_semantic_search_needed
            or not filters.query_text
            or len(filters.query_text.strip()) < 3
    ):
        logger.warning("Недостаточно описания сюжета для угадывания.")
        return {
            "parsed_filter": filters,
            "found_movies": [],
            "error_reason": "insufficient_information",
        }

    logger.info(f"Векторизация детального сюжета: '{filters.query_text}'")
    query_vector = await generate_query_embedding(filters.query_text)

    found_movies: list[MovieDTO] = await fetch_guessed_movie_hybrid(
        query_vector=query_vector,
        raw_query_text=filters.query_text,
    )

    logger.info(f"В состояние занесено {len(found_movies)} фильмов.")

    return {
        "parsed_filter": filters,
        "found_movies": found_movies,
        "error_reason": None,
    }


async def general_chat(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    raw_messages = state.get("messages") or []

    messages = [SystemMessage(content=prompts["SYSTEM_GENERAL_CHAT_PROMPT"])] + list(raw_messages)

    try:
        response = await llm.ainvoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        logger.error(f"Ошибка в general_chat: {e}")
        return {
            "final_response": "Привет! Я помогу найти фильм по описанию или подобрать классную подборку. Что бы ты хотел посмотреть?"
        }


async def synthesis_response_node(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    if state.get("error_reason") == "insufficient_information":
        return {
            "final_response": (
                "Хмм, ты дал слишком мало деталей о фильме!\n\n"
                "Расскажи хотя бы пару слов о сюжете, ключевой сцене, "
                "персонаже или о том, где происходило действие, и я обязательно его найду!"
            )
        }

    found_movies: list[MovieDTO] = state.get("found_movies", [])

    if not found_movies:
        return {
            "final_response": (
                "К сожалению, мне не удалось найти подходящие фильмы по твоему запросу.\n"
                "Попробуй немного изменить описание или снизить критерии поиска!"
            )
        }

    movies_context_list = []
    for m in found_movies:
        year = m.release_date[:4] if m.release_date else "Н/Д"
        rating = f"{m.vote_average:.1f}" if m.vote_average else "Н/Д"
        genres = m.genres if m.genres else "Н/Д"
        overview = m.overview if m.overview else "Нет описания"

        movies_context_list.append(
            f"ID: {m.id}\nНазвание: {m.title}\nГод: {year}\n"
            f"Рейтинг: {rating}\nЖанры: {genres}\nОписание: {overview}"
        )

    movies_context = "\n\n".join(movies_context_list)
    intent = state.get("intent", "recommend_movies")
    user_query = state["messages"][-1].content

    prompt = f"""
Запрос пользователя: "{user_query}"
Интент: {intent}

Найденные фильмы в базе данных:
{movies_context}

Сформируй красивый ответ для Telegram согласно инструкциям.
"""

    messages = [
        SystemMessage(content=prompts["SYSTEM_SYNTHESIS_PROMPT"]),
        SystemMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        logger.error(f"Ошибка при синтезе ответа: {e}")
        movie_titles = ", ".join([f"**{m.title}**" for m in found_movies])
        return {"final_response": f"Вот что мне удалось найти: {movie_titles}"}