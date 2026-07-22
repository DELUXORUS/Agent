from pydantic import BaseModel, Field

class TelegramMessageTask(BaseModel):
    user_id: int = Field(description="Telegram user ID")
    chat_id: int = Field(description="Telegram chat ID")
    message_id: int = Field(description="Message ID")
    prompt: str = Field(description="User`s prompt")
