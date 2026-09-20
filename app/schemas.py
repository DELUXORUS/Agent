from datetime import date
from pydantic import BaseModel, ConfigDict, Field


class TelegramMessageTask(BaseModel):
    user_id: int
    username: str
    chat_id: int
    message_id: int
    text: str

class TelegramCallbackTask(BaseModel):
    user_id: int
    chat_id: int
    message_id: int
    callback_query_id: str
    callback_data: str
    reply_markup: dict | None = None

class MovieDTO(BaseModel):
    id: int
    tmdb_id: int
    imdb_id: str | None = None
    title: str
    original_title: str | None = None
    original_language: str | None = None
    overview: str
    genres: list[str]
    actors: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    tagline: str | None = None
    release_date: date
    runtime: int = Field(gt=0)
    vote_average: float = Field(ge=0, le=10, allow_inf_nan=False)
    vote_count: int = Field(ge=0)
    keywords: list[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)
