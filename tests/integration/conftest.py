import os
from collections.abc import AsyncIterator
from datetime import date
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine

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

    pg_session.add_all([
        Movie(id=1, title="Orbit", release_date=date(2010, 1, 1),
              vote_average=7.0, embedding=vector(1, 0)),
        Movie(id=2, title="Orbit", release_date=date(2020, 12, 31),
              vote_average=8.0, embedding=vector(0.8, 0.6)),
        Movie(id=3, title="Moon station", release_date=date(2015, 6, 1),
              vote_average=8.0, embedding=vector(0.6, 0.8)),
        Movie(id=4, title="Village", release_date=date(2018, 6, 1),
              vote_average=9.0, embedding=vector(0, 1)),
        Movie(id=5, title="Old orbit", release_date=date(2009, 12, 31),
              vote_average=9.0, embedding=vector(1, 0)),
        Movie(id=6, title="Low rated orbit", release_date=date(2015, 6, 1),
              vote_average=6.9, embedding=vector(1, 0)),
        Movie(id=7, title="Unknown metadata", release_date=None,
              vote_average=None, embedding=vector(0.6, 0.8)),
        Movie(id=8, title="No embedding", release_date=date(2019, 6, 1),
              vote_average=8.5, embedding=None),
    ])
    # Insert parent records before the foreign-key references.
    await pg_session.flush()
    pg_session.add_all([
        UserMovieHistory(user_id=42, movie_id=1),
        UserMovieHistory(user_id=99, movie_id=2),
    ])
    await pg_session.flush()
    return pg_session
