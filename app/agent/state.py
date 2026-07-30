from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from app.schemas import MovieDTO

class MovieFilter(BaseModel):
    query_text: str | None = Field(
        default=None,
        description=(
            "Очищенное текстовое описание сюжета, атмосферы или темы для векторного поиска (ПЕРЕВОД НА АНГЛИЙСКИЙ). "
            "Например: 'space exploration survival on unknown planet'."
        ),
    )
    is_semantic_search_needed: bool = Field(
        default=True,
        description="Флаг: True, если пользователь описывает сюжет или тему; False, если запрос только по точным фильтрам (актер, жанр, рейтинг).",
    )

    title: str | None = Field(
        default=None,
        description="Конкретное название фильма, если пользователь явно назвал его.",
    )
    genre: str | None = Field(
        default=None,
        description="Жанр фильма на английском (например: Action, Sci-Fi, Drama, Comedy, Horror).",
    )
    credits: str | None = Field(
        default=None,
        description="Имя/фамилия актера или режиссера на английском (соответствует колонке credits в БД, например: 'Leonardo DiCaprio').",
    )
    keywords: str | None = Field(
        default=None,
        description="Ключевые слова или теги темы на английском (например: 'time travel', 'superhero', 'zombie').",
    )
    min_vote_average: float | None = Field(
        default=None,
        description="Минимальный рейтинг фильма от 1.0 до 10.0 (соответствует vote_average).",
    )
    release_date: int | None = Field(
        default=None,
        description="Год выпуска фильма, если указан конкретный год (например: 2010).",
    )

class IntentClassification(BaseModel):
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] = Field(
        description=(
            "Выбери 'guess_movie', если пользователь описывает сюжет конкретного фильма, который пытается вспомнить/угадать. "
            "Выбери 'recommend_movies', если пользователь просит посоветовать/подобрать список фильмов под жанр/настроение. "
            "Выбери 'general_chat' для остальных вопросов и приветствий."
        )
    )

class UnifiedParseResult(BaseModel):
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] = Field(
        description="Интент пользователя: guess_movie, recommend_movies или general_chat"
    )
    filter: MovieFilter = Field(
        default_factory=MovieFilter,
        description="Фильтры и англоязычный векторный контекст query_text"
    )

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # История сообщений
    user_id: int  # Telegram ID пользователя
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] | None
    parsed_filter: MovieFilter | None  # Результат работы Structured Output
    found_movies: list[MovieDTO]  # Список найденных объектов фильмов из Postgres
    final_response: str  # Готовый текст ответа для Telegram
    error_reason: str