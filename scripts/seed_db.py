import asyncio
from dataclasses import asdict
from typing import TYPE_CHECKING

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import EMBEDDING_DIMENSION
from app.db.models import Movie
from scripts.movie_dataset import (
    MovieSeedRecord,
    build_movie_embedding_text,
    build_movie_seed_records,
    merge_movie_datasets,
    prepare_and_filter_movies,
)

if TYPE_CHECKING:
    from app.services.embedder import EmbedderService

DEFAULT_SEED_BATCH_SIZE = 200


async def generate_movie_embeddings(
    movies: list[MovieSeedRecord],
    embedder: "EmbedderService",
) -> list[list[float]]:
    if not movies:
        return []

    semantic_movie_texts = [
        build_movie_embedding_text(movie)
        for movie in movies
    ]

    embeddings = await embedder.get_embeddings(
        semantic_movie_texts,
        batch_size=32,
    )

    if len(embeddings) != len(movies):
        raise ValueError(
            f"Expected {len(movies)} embeddings, "
            f"got {len(embeddings)}"
        )

    for index, embedding in enumerate(embeddings):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding at index {index}: "
                f"expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(embedding)}"
            )

    return embeddings


def build_movie_rows(
    movies: list[MovieSeedRecord],
    embeddings: list[list[float]],
) -> list[dict[str, object]]:
    """Prepare insert rows from records and their already validated vectors."""
    rows: list[dict[str, object]] = []
    for movie, embedding in zip(movies, embeddings, strict=True):
        row = asdict(movie)
        for field in ("genres", "actors", "directors", "keywords"):
            row[field] = list(row[field])
        row["embedding"] = list(embedding)
        rows.append(row)
    return rows


async def upsert_movies(
    session: AsyncSession,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return

    stmt = insert(Movie).values(rows)

    stmt = stmt.on_conflict_do_update(
        index_elements=[Movie.tmdb_id],
        set_={
            "imdb_id": stmt.excluded.imdb_id,
            "title": stmt.excluded.title,
            "normalized_title": stmt.excluded.normalized_title,
            "original_title": stmt.excluded.original_title,
            "normalized_original_title": stmt.excluded.normalized_original_title,
            "original_language": stmt.excluded.original_language,
            "overview": stmt.excluded.overview,
            "tagline": stmt.excluded.tagline,
            "genres": stmt.excluded.genres,
            "actors": stmt.excluded.actors,
            "directors": stmt.excluded.directors,
            "keywords": stmt.excluded.keywords,
            "release_date": stmt.excluded.release_date,
            "runtime": stmt.excluded.runtime,
            "vote_average": stmt.excluded.vote_average,
            "vote_count": stmt.excluded.vote_count,
            "embedding": stmt.excluded.embedding,
        },
    )

    await session.execute(stmt)
    await session.commit()


async def seed_movies(
    session: AsyncSession,
    movies: list[MovieSeedRecord],
    embedder: "EmbedderService",
    batch_size: int = DEFAULT_SEED_BATCH_SIZE,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    seeded_count = 0
    total_count = len(movies)

    for start in range(0, total_count, batch_size):
        batch = movies[start:start + batch_size]
        embeddings = await generate_movie_embeddings(batch, embedder)
        rows = build_movie_rows(batch, embeddings)
        await upsert_movies(session, rows)

        seeded_count += len(rows)
        print(f"Seeded movies: {seeded_count}/{total_count}")

    return seeded_count


async def main() -> None:
    from app.db.database import async_session_maker, engine, init_db
    from app.services.embedder import embedder
    from scripts.inspect_movie_dataset import load_raw_datasets

    movies, credits, keywords = load_raw_datasets()
    merged = merge_movie_datasets(movies, credits, keywords)
    prepared, _ = prepare_and_filter_movies(merged)
    movie_records = build_movie_seed_records(prepared)

    try:
        await init_db()
        async with async_session_maker() as session:
            seeded_count = await seed_movies(
                session,
                movie_records,
                embedder,
            )
    finally:
        await engine.dispose()

    print(f"Database seeding completed: {seeded_count} movies")


if __name__ == "__main__":
    asyncio.run(main())
