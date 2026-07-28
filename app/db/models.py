from datetime import date
from sqlalchemy import BigInteger, Text, Float, Date, String, ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    title: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    genres: Mapped[str | None] = mapped_column(String(255), nullable=True)
    release_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    vote_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    __table_args__ = (
        Index(
            "idx_movies_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",  # Алгоритм индекса
            postgresql_ops={"embedding": "vector_cosine_ops"},  # Косинусное расстояние
            postgresql_with={
                "m": 16,
                "ef_construction": 64,
            },  # Параметры графа HNSW
        ),
    )


class UserMovieHistory(Base):
    __tablename__ = "user_movies_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Telegram ID пользователя (например, 123456789)
    # Делаем index=True, чтобы мгновенно находить историю конкретного юзера
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    # Внешний ключ на ID фильма из таблицы "movies"
    movie_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    # watched_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())