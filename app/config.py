from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    BOT_TOKEN: str
    OPENROUTER_API_KEY: str
    RABBITMQ_URL: str
    EMBEDDER_MODEL_NAME: str
    LLM_MODEL_NAME: str
    OPENROUTER_URL: str

    LLM_STRICT_TEMPERATURE: float = 0.0  # Для классификации и парсинга JSON
    LLM_CREATIVE_TEMPERATURE: float = 0.7  # Для диалога и генерации ответов
    PROMPTS_PATH: str

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres_password"
    POSTGRES_DB: str = "agent_db"
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int = 5432


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        # 1. Если хост явно прописан в .env — берем его
        if self.POSTGRES_HOST:
            host = self.POSTGRES_HOST
        # 2. Если запущены внутри Docker-контейнера — подключаемся к сервису "postgres"
        elif os.path.exists("/.dockerenv"):
            host = "postgres"
        # 3. Если запускаем скрипт локально с ПК — подключаемся к "localhost"
        else:
            host = "localhost"

        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD}@{host}:"
            f"{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

settings = Settings()


