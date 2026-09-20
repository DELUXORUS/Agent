from sqlalchemy import (
    BigInteger, CheckConstraint, Date, Float,
    ForeignKey, Index, Integer, JSON, String,
    Text, UniqueConstraint,
)
from datetime import date

from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column
)
from pgvector.sqlalchemy import Vector

from app.constants import EMBEDDING_DIMENSION


class Base(DeclarativeBase):
    pass


def text_array_type():
    return ARRAY(Text()).with_variant(JSON(), "sqlite")


class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tmdb_id: Mapped[int] = mapped_column(Integer, nullable=False)
    imdb_id: Mapped[str | None] = mapped_column(String(16))

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    original_title: Mapped[str | None] = mapped_column(String(255))
    normalized_original_title: Mapped[str | None] = mapped_column(
        String(255)
    )
    original_language: Mapped[str | None] = mapped_column(String(10))

    overview: Mapped[str] = mapped_column(Text, nullable=False)
    tagline: Mapped[str | None] = mapped_column(Text)

    genres: Mapped[list[str]] = mapped_column(
        text_array_type(),
        nullable=False,
        default=list,
    )
    actors: Mapped[list[str]] = mapped_column(
        text_array_type(),
        nullable=False,
        default=list,
    )
    directors: Mapped[list[str]] = mapped_column(
        text_array_type(),
        nullable=False,
        default=list,
    )
    keywords: Mapped[list[str]] = mapped_column(
        text_array_type(),
        nullable=False,
        default=list,
    )

    release_date: Mapped[date] = mapped_column(Date, nullable=False)
    runtime: Mapped[int] = mapped_column(Integer, nullable=False)
    vote_average: Mapped[float] = mapped_column(Float, nullable=False)
    vote_count: Mapped[int] = mapped_column(Integer, nullable=False)

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION)
    )

    __table_args__ = (
        UniqueConstraint("tmdb_id", name="uq_movies_tmdb_id"),
        UniqueConstraint("imdb_id", name="uq_movies_imdb_id"),
        CheckConstraint(
            "runtime > 0",
            name="ck_movies_runtime_positive",
        ),
        CheckConstraint(
            "vote_average >= 0 AND vote_average <= 10",
            name="ck_movies_vote_average_range",
        ),
        CheckConstraint(
            "vote_count >= 0",
            name="ck_movies_vote_count_non_negative",
        ),
        Index(
            "idx_movies_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        Index("idx_movies_normalized_title", "normalized_title"),
        Index(
            "idx_movies_normalized_original_title",
            "normalized_original_title",
        ),
        Index(
            "idx_movies_genres_gin",
            "genres",
            postgresql_using="gin",
        ),
        Index(
            "idx_movies_actors_gin",
            "actors",
            postgresql_using="gin",
        ),
        Index(
            "idx_movies_directors_gin",
            "directors",
            postgresql_using="gin",
        ),
        Index("idx_movies_vote_average", "vote_average"),
        Index("idx_movies_release_date", "release_date"),
        Index("idx_movies_runtime", "runtime"),
    )


class UserMovieHistory(Base):
    __tablename__ = "user_movies_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    movie_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("movies.id", ondelete="CASCADE"),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "movie_id",
            name="uq_user_movies_history_user_movie",
        ),
    )
