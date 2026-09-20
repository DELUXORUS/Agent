import os
from collections.abc import AsyncIterator
from datetime import date
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

from app.core.text_normalization import normalize_search_text
from app.db.models import Base, Movie, UserMovieHistory


@pytest_asyncio.fixture
async def pg_engine() -> AsyncIterator[AsyncEngine]:
    database_url = os.environ.get("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("Set TEST_DATABASE_URL to run PostgreSQL integration tests")

    engine = create_async_engine(database_url)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def pg_session(pg_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """Create real tables in an isolated schema; roll back all DDL and data."""
    schema = f"test_{uuid4().hex}"
    async with pg_engine.connect() as connection:
        transaction = await connection.begin()
        try:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(text(f'SET LOCAL search_path TO "{schema}", public'))
            # Always create our tables, even if public contains tables of the same name.
            await connection.run_sync(lambda conn: Base.metadata.create_all(conn, checkfirst=False))
            async with AsyncSession(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="create_savepoint",
            ) as session:
                yield session
        finally:
            await transaction.rollback()


@pytest.fixture
def query_embedding() -> list[float]:
    """Synthetic query vector; no embedding model or network is needed."""
    return [1.0, 0.0] + [0.0] * 382


@pytest_asyncio.fixture
async def pg_catalog(pg_session: AsyncSession) -> AsyncSession:
    """Seed deterministic fictional movies and two users' viewing histories.

    Against query_embedding the closest matches are 1/5/6, then 2, then 3/7,
    then 4. Movie 8 has no vector. IDs 5 and 6 exercise hard-filter exclusions.
    """
    def vector(x: float, y: float) -> list[float]:
        return [x, y] + [0.0] * 382

    def movie(
        movie_id: int,
        title: str,
        release_date: date,
        vote_average: float,
        embedding: list[float] | None,
    ) -> Movie:
        return Movie(
            id=movie_id,
            tmdb_id=1000 + movie_id,
            imdb_id=f"tt{movie_id:07d}",
            title=title,
            normalized_title=normalize_search_text(title),
            overview=f"Overview for {title}",
            genres=["science fiction"],
            actors=[],
            directors=[],
            keywords=[],
            release_date=release_date,
            runtime=100,
            vote_average=vote_average,
            vote_count=100,
            embedding=embedding,
        )

    pg_session.add_all([
        movie(1, "Orbit", date(2010, 1, 1), 7.0, vector(1, 0)),
        movie(2, "Orbit", date(2020, 12, 31), 8.0, vector(0.8, 0.6)),
        movie(3, "Moon station", date(2015, 6, 1), 8.0, vector(0.6, 0.8)),
        movie(4, "Village", date(2018, 6, 1), 9.0, vector(0, 1)),
        movie(5, "Old orbit", date(2009, 12, 31), 9.0, vector(1, 0)),
        movie(6, "Low rated orbit", date(2015, 6, 1), 6.9, vector(1, 0)),
        movie(7, "Low metadata score", date(2000, 6, 1), 5.0, vector(0.6, 0.8)),
        movie(8, "No embedding", date(2019, 6, 1), 8.5, None),
    ])
    # Insert parent records before the foreign-key references.
    await pg_session.flush()
    pg_session.add_all([
        UserMovieHistory(user_id=42, movie_id=1),
        UserMovieHistory(user_id=99, movie_id=2),
    ])
    await pg_session.flush()
    return pg_session
