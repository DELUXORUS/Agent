from datetime import date

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.constants import EMBEDDING_DIMENSION
from app.db.models import Movie, UserMovieHistory
from app.db.operations import Operations
from scripts.seed_db import upsert_movies


@pytest.mark.asyncio
async def test_database_connection(pg_engine: AsyncEngine):
    async with pg_engine.connect() as connection:
        result = await connection.execute(text("SELECT 1"))
        assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_new_session_has_empty_tables(pg_session: AsyncSession):
    assert await pg_session.scalar(select(func.count()).select_from(Movie)) == 0
    assert await pg_session.scalar(select(func.count()).select_from(UserMovieHistory)) == 0
    schema = await pg_session.scalar(text("SELECT current_schema()"))
    assert schema.startswith("test_")
    index = await pg_session.scalar(text(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND indexname = 'idx_movies_embedding_hnsw'"
    ))
    assert "USING hnsw" in index


@pytest.mark.asyncio
async def test_catalog_contains_movies_vectors_and_user_history(pg_catalog: AsyncSession):
    assert await pg_catalog.scalar(select(func.count()).select_from(Movie)) == 8
    assert await pg_catalog.scalar(select(func.count()).select_from(UserMovieHistory)) == 2
    dimensions = await pg_catalog.scalar(text(
        "SELECT vector_dims(embedding) FROM movies WHERE id = 1"
    ))
    assert dimensions == 384
    assert await pg_catalog.scalar(select(Movie.embedding).where(Movie.id == 8)) is None
    history = (await pg_catalog.execute(
        select(UserMovieHistory.user_id, UserMovieHistory.movie_id).order_by(UserMovieHistory.user_id)
    )).all()
    assert history == [(42, 1), (99, 2)]


@pytest.mark.asyncio
async def test_get_movie_embeddings_returns_ids_and_skips_missing_vectors(
    pg_catalog: AsyncSession,
):
    embeddings = await Operations(pg_catalog).get_movie_embeddings([1, 2, 8, 999])

    assert set(embeddings) == {1, 2}
    assert all(len(embedding) == 384 for embedding in embeddings.values())


@pytest.mark.asyncio
async def test_history_insert_is_idempotent_per_user_and_movie(
    pg_catalog: AsyncSession,
):
    operations = Operations(pg_catalog)

    await operations.add_movies_to_user_history(user_id=777, movie_id=3)
    await operations.add_movies_to_user_history(user_id=777, movie_id=3)
    await operations.add_movies_to_user_history(user_id=888, movie_id=3)
    await operations.add_movies_to_user_history(user_id=777, movie_id=4)

    # Check all rows, not a set: a duplicate must fail this assertion.
    history = (await pg_catalog.execute(
        select(UserMovieHistory.user_id, UserMovieHistory.movie_id)
        .order_by(UserMovieHistory.user_id, UserMovieHistory.movie_id)
    )).all()

    assert history == [(42, 1), (99, 2), (777, 3), (777, 4), (888, 3)]


@pytest.mark.asyncio
async def test_movie_upsert_updates_existing_tmdb_movie(
    pg_session: AsyncSession,
):
    row = {
        "tmdb_id": 2649,
        "imdb_id": "tt0119174",
        "title": "The Game",
        "normalized_title": "the game",
        "original_title": "The Game",
        "normalized_original_title": "the game",
        "original_language": "en",
        "overview": "Original overview",
        "tagline": None,
        "genres": ["drama", "thriller"],
        "actors": ["michael douglas"],
        "directors": ["david fincher"],
        "keywords": ["game"],
        "release_date": date(1997, 9, 12),
        "runtime": 129,
        "vote_average": 7.5,
        "vote_count": 1000,
        "embedding": [0.1] * EMBEDDING_DIMENSION,
    }

    await upsert_movies(pg_session, [row])
    updated_row = {
        **row,
        "overview": "Updated overview",
        "vote_average": 8.0,
        "embedding": [0.2] * EMBEDDING_DIMENSION,
    }
    await upsert_movies(pg_session, [updated_row])

    stored_movie = await pg_session.scalar(
        select(Movie).where(Movie.tmdb_id == row["tmdb_id"])
    )
    assert stored_movie is not None
    assert stored_movie.overview == "Updated overview"
    assert stored_movie.vote_average == 8.0
    assert stored_movie.embedding == pytest.approx(updated_row["embedding"])
    assert await pg_session.scalar(select(func.count()).select_from(Movie)) == 1
