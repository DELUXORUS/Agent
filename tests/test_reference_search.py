"""Reference title/year lookup through the node, service and local SQL execution."""
from datetime import date
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.agent.nodes import resolve_references
from app.agent.routing import route_after_resolve_references
from app.agent.schemas import (
    Intent, IntRange, MovieQueryPlan, MovieReference, ReferenceResolutionStatus,
)
from app.db.models import Base, Movie
from app.db.operations import Operations
from app.services.movie_search import MovieSearchService


def make_movie(movie_id: int, title: str, release_date: date) -> Movie:
    return Movie(
        id=movie_id,
        tmdb_id=1000 + movie_id,
        imdb_id=f"tt{movie_id:07d}",
        title=title,
        normalized_title=title.casefold(),
        overview=f"Overview for {title}",
        genres=["science fiction"],
        actors=[],
        directors=[],
        keywords=[],
        release_date=release_date,
        runtime=120,
        vote_average=7.0,
        vote_count=100,
    )


@pytest.fixture
def database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all([
            make_movie(1, "Dune", date(1983, 12, 31)),
            make_movie(2, "Dune", date(1984, 1, 1)),
            make_movie(3, "Dune", date(1984, 12, 31)),
            make_movie(4, "Dune", date(1985, 1, 1)),
            make_movie(5, "Dune", date(2021, 10, 22)),
            make_movie(6, "Dune", date(2022, 1, 1)),
            make_movie(7, "Other title", date(1984, 6, 1)),
        ])
        session.commit()
        yield session
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("title,year,expected_ids", [
    ("Dune", None, [1, 2, 3, 4, 5, 6]),
    ("Dune", 1984, [2, 3]),
    ("Dune", 2021, [5]),
    ("Dune", 2000, []),
    ("Unknown", None, []),
    ("Unknown", 1984, []),
])
async def test_title_lookup_with_optional_year(database, title, year, expected_ids):
    session = Mock(execute=AsyncMock(side_effect=database.execute))
    movies = await Operations(session).find_movies_by_title(title, year=year)
    assert [movie.id for movie in movies] == expected_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("year,status,expected_id,candidate_ids", [
    (None, ReferenceResolutionStatus.AMBIGUOUS, None, [1, 2, 3, 4, 5, 6]),
    (1984, ReferenceResolutionStatus.AMBIGUOUS, None, [2, 3]),
    (2021, ReferenceResolutionStatus.RESOLVED, 5, []),
    (2000, ReferenceResolutionStatus.NOT_FOUND, None, []),
])
async def test_reference_year_reaches_database(
    database, year, status, expected_id, candidate_ids,
):
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.execute.side_effect = database.execute
    embedder = Mock(get_embedding=AsyncMock())
    service = MovieSearchService(Mock(return_value=session), embedder)
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        reference_movies=[MovieReference(query="Dune", exact_title=True, year=year)],
        # Recommendation years must not constrain the reference lookup.
        year=IntRange(min=2025),
    )
    update = await resolve_references({"query_plan": plan}, service)
    resolved = update["resolved_references"][0]
    assert resolved.status == status
    assert (resolved.movie.id if resolved.movie else None) == expected_id
    assert [movie.id for movie in resolved.candidates] == candidate_ids
    assert plan.year.min == 2025
    expected_route = (
        "search_movies" if status == ReferenceResolutionStatus.RESOLVED
        else "request_reference_clarification"
    )
    assert route_after_resolve_references(update) == expected_route
    embedder.get_embedding.assert_not_awaited()
    session.__aexit__.assert_awaited_once()
