from typing import Annotated, Literal, TypedDict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from app.schemas import MovieDTO
from enum import Enum


class IntentEnum(str, Enum):
    GUESS_MOVIE = "guess_movie"
    RECOMMEND_MOVIES = "recommend_movies"
    GENERAL_CHAT = "general_chat"

class MovieFilter(BaseModel):
    is_semantic_search_needed: bool = Field(
        default=True,
        description=(
            "True, если в запросе есть сюжет, атмосфера или просьба найти 'похожие на фильм X'. "
            "False, если пользователь указал только точные фильтры без темы/сюжета (например, 'посоветуй боевик с рейтингом от 8')."
        ),
    )
    query_text: Optional[str] = Field(
        default=None,
        description=(
            "Очищенный текст сюжета/атмосферы для векторного поиска (ОБЯЗАТЕЛЬНО НА АНГЛИЙСКОМ). Удали мусор ('привет', 'посоветуй'). "
            "QUERY EXPANSION: Если запрос 'похожие на [Фильм X]', сгенерируй детальное описание сюжета, тем, атмосферы и стиля Фильма X на английском. "
            "Если 'is_semantic_search_needed' == False — установи None."
        ),
    )
    title: Optional[str] = Field(
        default=None,
        description="Название конкретного фильма или фильма-ориентира, упомянутого пользователем (например, 'The Matrix', 'The Game'). Иначе None.",
    )
    genre: Optional[str] = Field(
        default=None,
        description=(
            "Жанр строго на АНГЛИЙСКОМ с заглавной буквы (Action, Adventure, Animation, Comedy, Crime, Documentary, "
            "Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, TV Movie, War, Western)."
        ),
    )
    credits: Optional[str] = Field(
        default=None,
        description=(
            "Имя/фамилия актера или режиссера на АНГЛИЙСКОМ, ТОЛЬКО если пользователь прямо ищет фильмы С ИХ УЧАСТИЕМ "
            "(например: 'Leonardo DiCaprio', 'Christopher Nolan'). "
            "ВНИМАНИЕ: НЕ заполняй это поле режиссером фильма-образца при запросах 'похожие на [Фильм X]'!"
        ),
    )
    keywords: Optional[str] = Field(
        default=None,
        description="1-2 точечных тега или ключевых слова на АНГЛИЙСКОМ (например: 'zombie', 'time travel', 'mafia'). Иначе None.",
    )
    min_vote_average: Optional[float] = Field(
        default=None,
        description="Минимальный рейтинг фильма числом от 1.0 до 10.0, если есть явное требование в запросе ('от 7.5'). Иначе None.",
    )
    release_date: Optional[int] = Field(
        default=None,
        description=(
            "Год выпуска фильма 4-значным числом, если пользователь явно затребовал конкретный год (например: 2010). "
            "ВНИМАНИЕ: НЕ заполняй это поле годом фильма-образца при запросах 'похожие на [Фильм X]'!"
        ),
    )


class UnifiedParseResult(BaseModel):
    intent: IntentEnum = Field(
        description=(
            "Намерение пользователя: "
            "'guess_movie' — пытается вспомнить конкретный фильм по описанию сюжета/сцены; "
            "'recommend_movies' — просит подобрать подборку фильмов по жанру, теме или сходству; "
            "'general_chat' — приветствие, оффтоп, вопрос про бота."
        )
    )
    filter: MovieFilter = Field(
        description="Спрогнозированный объект фильтрации на основе анализа запроса."
    )

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: int
    intent: Literal["guess_movie", "recommend_movies", "general_chat"] | None
    parsed_filter: MovieFilter | None
    found_movies: list[MovieDTO]
    final_response: str
    error_reason: str