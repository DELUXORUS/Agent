from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List

class TelegramMessageTask(BaseModel):
    user_id: int = Field(description="Telegram user ID")
    username: str = Field(description="Telegram username")
    chat_id: int = Field(description="Telegram chat ID")
    message_id: int = Field(description="Message ID")
    text: str = Field(description="User`s text")

class MovieDTO(BaseModel):
    id: int
    title: str
    overview: Optional[str] = None
    genres: Optional[str] = None
    vote_average: Optional[float] = None
    release_date: Optional[str] = None
    poster_path: Optional[str] = None

    # Это ключевой флаг в Pydantic V2!
    # Он позволяет автоматически конвертировать модель SQLAlchemy в Pydantic DTO
    model_config = ConfigDict(from_attributes=True)
