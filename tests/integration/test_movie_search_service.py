from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.movie_search import MovieSearchService
from app.services.schemas import MovieSearchParams


def make_session_factory(
    session: AsyncSession,
) -> Callable[[], AsyncIterator[AsyncSession]]:
    @asynccontextmanager
    async def session_factory():
        yield session

    return session_factory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "expected_ids"),
    [
        (42, [5, 6, 2, 3]),
        (99, [1, 5, 6, 3]),
    ],
)
async def test_search_excludes_only_current_users_history(
    pg_catalog: AsyncSession,
    query_embedding: list[float],
    user_id: int,
    expected_ids: list[int],
):
    embedder = Mock(
        get_embedding=AsyncMock(return_value=query_embedding)
    )
    service = MovieSearchService(
        make_session_factory(pg_catalog),
        embedder,
    )

    movies = await service.search_recommendations(
        user_id=user_id,
        params=MovieSearchParams(semantic_query="space", limit=4),
    )

    assert [movie.id for movie in movies] == expected_ids
    embedder.get_embedding.assert_awaited_once_with("space")


@pytest.mark.asyncio
async def test_search_merges_explicit_and_watched_exclusions(
    pg_catalog: AsyncSession,
    query_embedding: list[float],
):
    embedder = Mock(
        get_embedding=AsyncMock(return_value=query_embedding)
    )
    service = MovieSearchService(
        make_session_factory(pg_catalog),
        embedder,
    )
    params = MovieSearchParams(
        semantic_query="space",
        excluded_movie_ids=[5, 6],
        limit=3,
    )

    movies = await service.search_recommendations(
        user_id=42,
        params=params,
    )

    assert [movie.id for movie in movies] == [2, 3, 7]
    assert params.excluded_movie_ids == [5, 6]


@pytest.mark.asyncio
async def test_search_without_semantic_query_uses_rating_order(
    pg_catalog: AsyncSession,
):
    embedder = Mock(get_embedding=AsyncMock())
    service = MovieSearchService(
        make_session_factory(pg_catalog),
        embedder,
    )
    params = MovieSearchParams(
        year_min=2010,
        year_max=2020,
        rating_min=7.0,
        rating_max=8.0,
        limit=5,
    )

    movies = await service.search_recommendations(
        user_id=42,
        params=params,
    )

    assert [movie.id for movie in movies] == [2, 3]
    embedder.get_embedding.assert_not_awaited()
