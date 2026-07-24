from pydantic import BaseModel, Field

class TelegramMessageTask(BaseModel):
    user_id: int = Field(description="Telegram user ID")
    username: str = Field(description="Telegram username")
    chat_id: int = Field(description="Telegram chat ID")
    message_id: int = Field(description="Message ID")
    text: str = Field(description="User`s text")
