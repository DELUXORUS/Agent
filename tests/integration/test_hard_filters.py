from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import select

from app.db.filters import MovieFilters, apply_movie_filters
from app.db.models import Movie
from app.services.movie_search import MovieSearchService
from app.services.schemas import MovieSearchParams


async def configure_catalog(session):
    movies = (await session.scalars(select(Movie))).all()
    for movie in movies:
        movie.genres = []
        movie.actors = []
        movie.directors = []
        movie.runtime = 100
    movies_by_id = {movie.id: movie for movie in movies}
    for field in ("genres", "actors", "directors"):
        setattr(movies_by_id[1], field, ["a", "b"])
        setattr(movies_by_id[2], field, ["a"])
        setattr(movies_by_id[3], field, ["b"])
    movies_by_id[1].runtime = 90
    movies_by_id[2].runtime = 120
    movies_by_id[3].runtime = 121
    await session.flush()


@pytest.mark.asyncio
@pytest.mark.parametrize("entity", ["genres", "actors", "directors"])
@pytest.mark.parametrize("included,excluded,expected", [
    (["a", "b"], [], [1]),
    (["a"], ["b"], [2]),
    ([], ["a", "b"], [4, 5, 6, 7, 8]),
    (["a"], ["a"], []),
    (["missing"], [], []),
    ([], [], [1, 2, 3, 4, 5, 6, 7, 8]),
])
async def test_array_filters(pg_catalog, entity, included, excluded, expected):
    await configure_catalog(pg_catalog)
    filters = MovieFilters(**{
        f"included_{entity}": included,
        f"excluded_{entity}": excluded,
    })
    stmt = apply_movie_filters(select(Movie), filters).order_by(Movie.id)
    assert [movie.id for movie in await pg_catalog.scalars(stmt)] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("minimum,maximum,expected", [
    (90, 120, [1, 2, 4, 5, 6, 7, 8]),
    (120, None, [2, 3]),
    (None, 90, [1]),
    (121, 90, []),
])
async def test_runtime_boundaries(pg_catalog, minimum, maximum, expected):
    await configure_catalog(pg_catalog)
    stmt = apply_movie_filters(select(Movie), MovieFilters(
        runtime_min=minimum, runtime_max=maximum,
    )).order_by(Movie.id)
    assert [movie.id for movie in await pg_catalog.scalars(stmt)] == expected


@pytest.mark.asyncio
async def test_service_normalizes_combined_filters_before_vector_limit(pg_catalog, query_embedding):
    await configure_catalog(pg_catalog)

    @asynccontextmanager
    async def sessions():
        yield pg_catalog

    embedder = Mock(get_embedding=AsyncMock(return_value=query_embedding))
    service = MovieSearchService(sessions, embedder)
    params = MovieSearchParams(
        genres=[" Ａ ", "a", " "], actors=[" A "], directors=["a"],
        excluded_genres=[" B "], excluded_actors=["b"], excluded_directors=["B"],
        runtime_min=90, runtime_max=120, year_min=2010, rating_min=7,
        semantic_query="space", candidate_limit=1,
    )
    movies = await service.search_recommendations(0, params)
    assert [movie.id for movie in movies] == [2]
    assert params.genres == [" Ａ ", "a", " "]
