from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):
    BOT_TOKEN: str
    RABBITMQ_URL: str
    # POSTGRES_URL: str
    OPENROUTER_API_KEY: str

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


