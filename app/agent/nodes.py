import logging
from typing import Any
from app.agent.state import AgentState, IntentClassification
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from app.agent.tools import (
    generate_query_embedding,
    fetch_recommended_movies,
    fetch_guessed_movie_hybrid
)
from app.agent.state import MovieFilter


logger = logging.getLogger("uvicorn")

SYSTEM_INTENT_PROMPT = """
Ты — профессиональный AI-классификатор интентов в сервисе поиска и рекомендаций кино.
Твоя единственная задача — проанализировать сообщение пользователя и точно определить его намерение.

Варианты интентов:
1. `guess_movie` — пользователь пытается вспомнить/угадать КОНКРЕТНЫЙ фильм по его сюжету, сцене, актерам или описанию.
   Примеры:
   - "Как называется фильм где чел застрял на Марсе?"
   - "Фильм где главный герой видит мертвых людей"
   - "Ищу кинчик там короче динозавры вырвались из парка"

2. `recommend_movies` — пользователь просит ПОДОБРАТЬ или ПОСОВЕТОВАТЬ список фильмов под критерии/жанр/настроение.
   Примеры:
   - "Посоветуй хороший боевик на вечер"
   - "Что посмотреть из фантастики с рейтингом от 8?"
   - "Посоветуй фильмы похожие на Интерстеллар"

3. `general_chat` — приветствия, вопросы о боте, благодарности или оффтоп.
   Примеры:
   - "Привет, кто ты?"
   - "Спасибо за подборку!"
   - "Какая сегодня погода?"
"""

SYSTEM_RECOMMEND_PARSER_PROMPT = """
Ты — AI-парсер. Твоя задача — проанализировать запрос пользователя и выделить параметры поиска.

1. `is_semantic_search_needed`: Установи True, если пользователь описывает сюжет, тему или атмосферу фильма (например: "про космос", "одиночество", "побег из тюрьмы"). 
   Установи False, если запрос состоит только из фильтров (например: "посоветуй боевик", "фильмы с рейтингом от 8").
2. `query_text`: Убери приветствия ("Привет", "Посоветуй"), удали упоминания рейтингов/жанров. 
   Оставь ТОЛЬКО ключевые слова сюжета и ПЕРЕВЕДИ ИХ НА АНГЛИЙСКИЙ (например: "space exploration survival"). Если is_semantic_search_needed=False — оставь None.
3. `genre`: Жанр на английском с заглавной буквы (Action, Sci-Fi, Drama, Comedy и т.д.), если явно указан. Иначе None.
4. `min_vote_average`: Число от 1.0 до 10.0, если есть требование по рейтингу. Иначе None.
"""

SYSTEM_GUESS_PARSER_PROMPT = """
Ты — AI-парсер описания сюжета для задачи угадывания конкретного фильма.
Пользователь описывает фильм, название которого он забыл, пытаясь вспомнить сюжет, сценарий, персонажей, ключевые сцены или локации.

Твоя задача — извлечь максимум смысловой информации для векторного и полнотекстового поиска:

1. `query_text`: 
   - Удали все приветствия, вежливые обращения и вводные фразы ("Привет", "Как называется фильм", "Помоги вспомнить", "Забыл название там короче").
   - Уделай внимание уникальным деталям: что делает главный герой, где происходят события, какие есть запоминающиеся предметы, концовка или повороты сюжета.
   - ПЕРЕВЕДИ полученное описание на АНГЛИЙСКИЙ ЯЗЫК и выдели ключевые сущности (например: "man trapped on mars farming potatoes astronaut survival").
2. `is_semantic_search_needed`: ВСЕГДА устанавливай True, так как поиск фильма по сюжету всегда требует семантического векторного поиска.
3. `genre`: ВСЕГДА оставляй None. Не пытайся угадывать жанр, чтобы случайно не ограничить поиск, если пользователь ошибся.
4. `min_vote_average`: ВСЕГДА оставляй None. Ограничения по рейтингу не нужны для поиска точного фильма.

Пример:
Вход: "Привет! Ищу фильм где мужик застрял во временной петле в одном дне и каждый день слушает будильник с песней I Got You Babe"
Результат query_text: "man trapped in time loop reliving same day alarm clock song I Got You Babe"
"""

SYSTEM_SYNTHESIS_PROMPT = """
Ты — дружелюбный и эрудированный кинокритик/ассистент, помогающий людям находить классное кино.

Твоя задача — составить красивый, вежливый и структурированный ответ для Telegram на основе найденных в базе данных фильмов.

Инструкции по оформлению:
1. Используй Markdown: выделяй названия фильмов **жирным шрифтом**, указывай год выпуска и рейтинг (например: ⭐ 8.2).
2. Давай короткое (1-2 предложения) увлекательное описание к каждому фильму без спойлеров.
3. Если интент 'guess_movie' — напиши в стиле: "Кажется, ты ищешь фильм **[Название]** ([Год])! Вот его сюжет: ..."
4. Если интент 'recommend_movies' — оформи красиво нумерованный список из найденных фильмов.
5. Завершай ответ легким призывом к диалогу или приятным пожеланием просмотра.
"""

SYSTEM_GENERAL_CHAT_PROMPT = """
Ты — вежливый AI-ассистент сервиса поиска и рекомендаций фильмов.
Отвечай коротко, дружелюбно и с юмором. 
Если пользователь приветствует тебя — поприветствуй в ответ и коротко расскажи, что ты умеешь (угадывать фильмы по описанию сюжета и подбирать подборки по жанрам/рейтингу).
"""


async def intent_classification_node(state: AgentState, llm: ChatOpenAI) -> dict [str, Any]:
    logger.info("Классификация...")

    structured_llm = llm.with_structured_output(IntentClassification)
    messages = [SystemMessage(content=SYSTEM_INTENT_PROMPT)] + state["messages"]

    try:
        result: IntentClassification = await structured_llm.ainvoke(messages)
        logger.info(f"RAG Intent Classified: {result.intent}")

        return {"intent": result.intent}

    except Exception as e:
        logger.error(
            f"Ошибка в intent_classifier_node: {e}. Фолбэк на general_chat"
        )

        return {"intent": "general_chat"}


async def search_for_recomended_node(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:
    last_user_message = state["messages"][-1].content

    structured_llm = llm.with_structured_output(MovieFilter)
    messages = ([SystemMessage(content=SYSTEM_RECOMMEND_PARSER_PROMPT)] +
                state["messages"])

    try:
        filters: MovieFilter = await structured_llm.ainvoke(messages)
    except Exception as e:
        logger.error(f"Ошибка при парсинге запроса: {e}")
        filters = MovieFilter(
            is_semantic_search_needed=True,
            query_text=last_user_message,  # Фолбэк на исходное сообщение
        )

    query_vector = None
    if filters.is_semantic_search_needed and filters.query_text:
        logger.info(f"Векторизация очищенного текста: '{filters.query_text}'")
        query_vector = generate_query_embedding(filters.query_text)
    else:
        logger.info("Поиск по точным SQL-фильтрам (без вектора)")

    found_movies = await fetch_recommended_movies(
        user_id=state['user_id'],
        query_vector=query_vector,
        limit=3,
        min_vote_average=filters.min_vote_average,
        genre=filters.genre,
    )

    logger.info(f"В состояние занесено {len(found_movies)} фильмов.")

    return {
        "parsed_filter": filters,
        "found_movies": found_movies,
    }


async def search_for_guess_node(
    state: AgentState, llm: ChatOpenAI
) -> dict[str, Any]:
    last_user_message = state["messages"][-1].content

    structured_llm = llm.with_structured_output(MovieFilter)
    # Исправлено: передаем промпт для GUESS, а не RECOMMEND
    messages = [SystemMessage(content=SYSTEM_GUESS_PARSER_PROMPT)] + state[
        "messages"
    ]

    try:
        filters: MovieFilter = await structured_llm.ainvoke(messages)
    except Exception as e:
        logger.error(f"Ошибка при парсинге запроса: {e}")
        filters = MovieFilter(
            is_semantic_search_needed=True,
            query_text=last_user_message,
        )

    # Защита от слишком короткого/пустого описания
    if (
        not filters.is_semantic_search_needed
        or not filters.query_text
        or len(filters.query_text.strip()) < 3
    ):
        logger.warning(
            "⚠Недостаточно описания сюжета для угадывания."
        )
        return {
            "parsed_filter": filters,
            "found_movies": [],
            "error_reason": "insufficient_information",
        }

    logger.info(f"Векторизация детального сюжета: '{filters.query_text}'")
    query_vector = generate_query_embedding(filters.query_text)

    # Исправлено: вызываем гибридный инструмент для точного угадывания
    found_movies = await fetch_guessed_movie_hybrid(
        query_vector=query_vector,
        raw_query_text=last_user_message,
    )

    logger.info(f"В состояние занесено {len(found_movies)} фильмов.")

    return {
        "parsed_filter": filters,
        "found_movies": found_movies,
        "error_reason": None,
    }


async def general_chat(state: AgentState, llm: ChatOpenAI) -> dict[str, Any]:

    messages = [SystemMessage(content=SYSTEM_GENERAL_CHAT_PROMPT)] + state[
        "messages"
    ]

    try:
        response = await llm.ainvoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        logger.error(f"Ошибка в general_chat: {e}")
        return {
            "final_response": "Привет! Я помогу найти фильм по описанию или подобрать классную подборку. Что бы ты хотел посмотреть?"
        }


async def synthesis_response_node(
    state: AgentState, llm: ChatOpenAI
) -> dict[str, Any]:
    # 1. Если сработал Guardrail в угадывании (слишком мало информации)
    if state.get("error_reason") == "insufficient_information":
        return {
            "final_response": (
                "Хмм, ты дал слишком мало деталей о фильме!\n\n"
                "Расскажи хотя бы пару слов о сюжете, ключевой сцене, "
                "персонаже или о том, где происходило действие, и я обязательно его найду!"
            )
        }

    found_movies = state.get("found_movies", [])

    # 2. Если поиск не дал результатов
    if not found_movies:
        return {
            "final_response": (
                "К сожалению, мне не удалось найти подходящие фильмы по твоему запросу.\n"
                "Попробуй немного изменить описание или снизить критерии поиска!"
            )
        }

    # 3. Подготавливаем контекст из найденных объектов фильмов для LLM
    movies_context = "\n\n".join(
        [
            f"ID: {m.id}\nНазвание: {m.title}\nГод: {getattr(m, 'release_date', 'Н/Д')}\n"
            f"Рейтинг: {getattr(m, 'vote_average', 'Н/Д')}\nЖанры: {getattr(m, 'genres', 'Н/Д')}\n"
            f"Описание: {getattr(m, 'overview', 'Нет описания')}"
            for m in found_movies
        ]
    )

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
        SystemMessage(content=SYSTEM_SYNTHESIS_PROMPT),
        SystemMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        logger.error(f"Ошибка при синтезе ответа: {e}")
        # Фолбэк из простого форматирования без участия LLM
        movie_titles = ", ".join([f"**{m.title}**" for m in found_movies])
        return {
            "final_response": f"Вот что мне удалось найти: {movie_titles}"
        }