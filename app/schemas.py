from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

class TelegramMessageTask(BaseModel):
    user_id: int = Field(description="Telegram user ID")
    username: str = Field(description="Telegram username")
    chat_id: int = Field(description="Telegram chat ID")
    message_id: int = Field(description="Message ID")
    text: str = Field(description="User`s text")

class MovieDTO(BaseModel):
    id: int
    title: str
    overview: str | None = None
    genres: str | None = None
    credits: str | None = None
    tagline: str | None = None
    release_date: str | None = None  # <-- Здесь всегда будет строка
    vote_average: float | None = None
    keywords: str | None = None

    # Это ключевой флаг в Pydantic V2!
    # Он позволяет автоматически конвертировать модель SQLAlchemy в Pydantic DTO
    model_config = ConfigDict(from_attributes=True)

    @field_validator("release_date", mode="before")
    @classmethod
    def convert_date_to_str(cls, v):
        if isinstance(v, date):
            return v.strftime("%Y-%m-%d")
        return v
