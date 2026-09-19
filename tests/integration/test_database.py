import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.db.models import Movie, UserMovieHistory
from app.db.operations import Operations


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
