import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.filters import MovieFilters
from app.db.operations import Operations


@pytest.mark.asyncio
async def test_search_by_year_and_rating(pg_catalog: AsyncSession):
    operations = Operations(pg_catalog)
    filters = MovieFilters(
        year_min=2010,
        year_max=2020,
        rating_min=7.0,
        rating_max=8.0,
    )

    movies = await operations.search_movies(
        filters=filters,
        query_embedding=None,
        limit=10,
    )

    # Rating descending, then ID ascending for equal ratings.
    # Dates and ratings at the inclusive boundaries must remain in the result.
    assert [movie.id for movie in movies] == [2, 3, 1]


@pytest.mark.asyncio
async def test_vector_search_orders_by_cosine_similarity(
    pg_catalog: AsyncSession,
    query_embedding: list[float],
):
    operations = Operations(pg_catalog)

    movies = await operations.search_movies(
        filters=MovieFilters(),
        query_embedding=query_embedding,
        limit=5,
    )

    # IDs 1, 5 and 6 have the same vector as the query; the ID is the tie-breaker.
    # ID 2 is closer than ID 3. A movie without an embedding cannot be ranked.
    assert [movie.id for movie in movies] == [1, 5, 6, 2, 3]
    assert all(movie.id != 8 for movie in movies)


@pytest.mark.asyncio
async def test_hard_filters_are_applied_before_vector_limit(
    pg_catalog: AsyncSession,
    query_embedding: list[float],
):
    operations = Operations(pg_catalog)
    filters = MovieFilters(
        year_min=2010,
        year_max=2020,
        rating_min=7.0,
        rating_max=8.0,
    )

    movies = await operations.search_movies(
        filters=filters,
        query_embedding=query_embedding,
        limit=2,
    )

    # IDs 5 and 6 are very close to the query but fail year/rating constraints.
    # PostgreSQL filters them first and then takes the two closest valid movies.
    assert [movie.id for movie in movies] == [1, 2]
