from typing import Annotated, Any, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class MovieFilter(BaseModel):
    query_text: str = Field(
        description="Очищенное текстовое описание сюжета/запроса для поиска векторов (на английском)."
    )
    genre: str | None = Field(
        default=None,
        description="Жанр фильма (только для рекомендаций, например: Action, Sci-Fi, Drama).",
    )
    min_vote_average: float | None = Field(
        default=None,
        description="Минимальный рейтинг фильма от 1.0 до 10.0 (только для рекомендаций).",
    )
    is_semantic_search_needed: bool = Field(
        default=None,
        description="Флаг для определения необходимости создания вектора смысла для запроса.",
    )

class IntentClassification(BaseModel):
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] = Field(
        description=(
            "Выбери 'guess_movie', если пользователь описывает сюжет конкретного фильма, который пытается вспомнить/угадать. "
            "Выбери 'recommend_movies', если пользователь просит посоветовать/подобрать список фильмов под жанр/настроение. "
            "Выбери 'general_chat' для остальных вопросов и приветствий."
        )
    )

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # История сообщений
    user_id: int  # Telegram ID пользователя
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] | None
    parsed_filter: MovieFilter | None  # Результат работы Structured Output
    found_movies: list[Any]  # Список найденных объектов фильмов из Postgres
    final_response: str  # Готовый текст ответа для Telegram
    error_reason: str