"""Local search checks; PostgreSQL vector execution is not simulated by SQLite."""
from datetime import date
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session

from app.agent.nodes import search_movies
from app.agent.schemas import FloatRange, IntRange, Intent, MovieQueryPlan
from app.db.filters import MovieFilters, apply_movie_filters
from app.db.mappers import build_movie_filters
from app.db.models import Base, Movie, UserMovieHistory
from app.db.operations import Operations
from app.services.movie_search import MovieSearchService
from app.services.schemas import MovieSearchParams


@pytest.fixture
def database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([
            Movie(id=1, title="Lower boundary", release_date=date(2010, 1, 1), vote_average=7),
            Movie(id=2, title="Upper boundary", release_date=date(2020, 12, 31), vote_average=8),
            Movie(id=3, title="Too old", release_date=date(2009, 12, 31), vote_average=9),
            Movie(id=4, title="Too new", release_date=date(2021, 1, 1), vote_average=9),
            Movie(id=5, title="Low rating", release_date=date(2015, 1, 1), vote_average=6.9),
            Movie(id=6, title="High rating", release_date=date(2015, 1, 1), vote_average=8.1),
            Movie(id=7, title="Unknown", release_date=None, vote_average=None),
            Movie(id=8, title="Same rating", release_date=date(2015, 1, 1), vote_average=8),
            Movie(id=9, title="Zero rating", release_date=date(2015, 1, 1), vote_average=0),
            UserMovieHistory(user_id=42, movie_id=2, id=1),
            UserMovieHistory(user_id=99, movie_id=1, id=2),
        ])
        session.commit()
        yield session
    engine.dispose()


def test_filters_keep_inclusive_boundaries_and_exclude_ids(database):
    filters = MovieFilters(year_min=2010, year_max=2020, rating_min=7, rating_max=8,
                           excluded_movie_ids=[8])
    stmt = apply_movie_filters(select(Movie), filters).order_by(Movie.id)
    assert [m.id for m in database.scalars(stmt)] == [1, 2]


def test_empty_filters_do_not_limit_or_sort(database):
    stmt = select(Movie)
    assert apply_movie_filters(stmt, MovieFilters()) is stmt
    assert len(database.scalars(stmt).all()) == 9


def test_zero_rating_is_a_constraint(database):
    stmt = apply_movie_filters(select(Movie), MovieFilters(rating_max=0))
    assert [m.id for m in database.scalars(stmt)] == [9]


@pytest.mark.parametrize("field,value", [
    ("runtime_min", 90), ("runtime_max", 120),
    ("included_genres", ["Drama"]), ("excluded_genres", ["Horror"]),
    ("included_actors", ["Actor"]), ("excluded_actors", ["Actor"]),
    ("included_directors", ["Director"]), ("excluded_directors", ["Director"]),
])
def test_unsupported_filters_are_not_silently_ignored(field, value):
    with pytest.raises(NotImplementedError):
        apply_movie_filters(select(Movie), MovieFilters(**{field: value}))


def test_mapper_preserves_constraints_without_sharing_lists():
    params = MovieSearchParams(year_min=2010, year_max=2020, rating_min=7, rating_max=8,
                               runtime_min=90, runtime_max=120, genres=["Drama"],
                               excluded_genres=["Horror"], actors=["Actor"],
                               directors=["Director"], excluded_movie_ids=[2])
    filters = build_movie_filters(params)
    assert filters == MovieFilters(year_min=2010, year_max=2020, rating_min=7, rating_max=8,
                                   runtime_min=90, runtime_max=120, included_genres=["Drama"],
                                   excluded_genres=["Horror"], included_actors=["Actor"],
                                   included_directors=["Director"], excluded_movie_ids=[2])
    filters.excluded_movie_ids.append(3)
    filters.included_genres.append("Comedy")
    assert params.excluded_movie_ids == [2]
    assert params.genres == ["Drama"]


@pytest.mark.asyncio
async def test_db_search_filters_before_limit_and_breaks_rating_ties(database):
    session = Mock(execute=AsyncMock(side_effect=database.execute))
    movies = await Operations(session).search_movies(
        MovieFilters(year_min=2010, year_max=2020, rating_max=8), None, 2
    )
    assert [m.id for m in movies] == [2, 8]
    assert movies[0].release_date == "2020-12-31"


@pytest.mark.asyncio
async def test_search_without_matches_returns_empty_list(database):
    session = Mock(execute=AsyncMock(side_effect=database.execute))
    assert await Operations(session).search_movies(MovieFilters(rating_min=10), None, 5) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_invalid_limit_does_not_execute_query(limit):
    session = Mock(execute=AsyncMock())
    with pytest.raises(ValueError, match="limit"):
        await Operations(session).search_movies(MovieFilters(), None, limit)
    session.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_vector_query_uses_cosine_order_and_same_constraints():
    result = Mock()
    result.scalars.return_value.all.return_value = []
    session = Mock(execute=AsyncMock(return_value=result))
    await Operations(session).search_movies(
        MovieFilters(rating_min=7, excluded_movie_ids=[2]), [0.1] * 384, 3
    )
    stmt = session.execute.call_args.args[0]
    sql = str(stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}))
    assert "movies.vote_average >= 7" in sql
    assert "NOT IN (2)" in sql
    assert "movies.embedding IS NOT NULL" in sql
    assert "ORDER BY movies.embedding <=>" in sql
    assert "LIMIT 3" in sql


@pytest.mark.asyncio
async def test_node_service_and_db_search_work_together(database):
    session = AsyncMock()
    session.execute.side_effect = database.execute
    session.__aenter__.return_value = session
    factory = Mock(return_value=session)
    embedder = Mock(get_embedding=AsyncMock())
    service = MovieSearchService(factory, embedder)
    state = {"request_id": "test", "user_id": 42, "user_query": "Films from 2010 to 2020",
             "query_plan": MovieQueryPlan(intent=Intent.RECOMMEND_MOVIES,
                                          year=IntRange(min=2010, max=2020),
                                          rating=FloatRange(min=7, max=8), limit=2)}
    result = await search_movies(state, service)
    # User 42 watched movie 2; user 99's history must not exclude movie 1.
    assert [m.id for m in result["candidates"]] == [8, 1]
    embedder.get_embedding.assert_not_awaited()
    session.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_semantic_search_passes_embedding_and_merges_exclusions(monkeypatch):
    session = AsyncMock()
    session.__aenter__.return_value = session
    factory = Mock(return_value=session)
    embedding = [0.1] * 384
    embedder = Mock(get_embedding=AsyncMock(return_value=embedding))
    operations = Mock(get_watched_movie_ids=AsyncMock(return_value=[2, 3]),
                      search_movies=AsyncMock(return_value=[]))
    monkeypatch.setattr("app.services.movie_search.Operations", Mock(return_value=operations))
    params = MovieSearchParams(semantic_query="space", excluded_movie_ids=[1, 2], limit=3)
    assert await MovieSearchService(factory, embedder).search_recommendations(42, params) == []
    embedder.get_embedding.assert_awaited_once_with("space")
    operations.get_watched_movie_ids.assert_awaited_once_with(42)
    kwargs = operations.search_movies.call_args.kwargs
    assert kwargs["query_embedding"] == embedding
    assert kwargs["limit"] == 3
    assert set(kwargs["filters"].excluded_movie_ids) == {1, 2, 3}
    assert params.excluded_movie_ids == [1, 2]


@pytest.mark.asyncio
async def test_embedding_failure_does_not_open_database():
    factory = Mock()
    embedder = Mock(get_embedding=AsyncMock(side_effect=RuntimeError("embedding failed")))
    with pytest.raises(RuntimeError, match="embedding failed"):
        await MovieSearchService(factory, embedder).search_recommendations(
            42, MovieSearchParams(semantic_query="space")
        )
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_database_failure_releases_session():
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.execute.side_effect = RuntimeError("database failed")
    service = MovieSearchService(Mock(return_value=session), Mock())
    with pytest.raises(RuntimeError, match="database failed"):
        await service.search_recommendations(42, MovieSearchParams())
    session.__aexit__.assert_awaited_once()
