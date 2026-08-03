from sqlalchemy import (
    BigInteger, Text, Float, String,
    ForeignKey, Index, Integer, Date
)
from datetime import date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass

class Movie(Base):
    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    overview: Mapped[str | None] = mapped_column(Text)
    genres: Mapped[str | None] = mapped_column(String)
    credits: Mapped[str | None] = mapped_column(Text)
    release_date: Mapped[date | None] = mapped_column(Date)
    vote_average: Mapped[float | None] = mapped_column(Float)
    keywords: Mapped[str | None] = mapped_column(Text)
    tagline: Mapped[str | None] = mapped_column(Text)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))

    __table_args__ = (
        Index(
            "idx_movies_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
        Index("idx_movies_genres", "genres"),
        Index("idx_movies_vote_average", "vote_average"),
        Index("idx_movies_release_date", "release_date"),
    )

class UserMovieHistory(Base):
    __tablename__ = "user_movies_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    movie_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
