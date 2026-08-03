from datetime import date
from pydantic import (
    BaseModel, ConfigDict,
    Field, field_validator
)

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
    title: str
    overview: str | None = None
    genres: str | None = None
    credits: str | None = None
    tagline: str | None = None
    release_date: str | None = None
    vote_average: float | None = None
    keywords: str | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("release_date", mode="before")
    @classmethod
    def convert_date_to_str(cls, v):
        if isinstance(v, date):
            return v.strftime("%Y-%m-%d")
        return v
