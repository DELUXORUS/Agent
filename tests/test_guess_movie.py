import pytest

from app.agent.nodes import request_guess_movie_unavailable
from app.agent.routing import route_after_parse
from app.agent.schemas import EntityFilter, Intent, MovieQueryPlan, MovieReference


def test_guess_movie_routes_to_unavailable_response():
    plan = MovieQueryPlan(
        intent=Intent.GUESS_MOVIE,
        reference_movies=[
            MovieReference(
                query="фильм Нолана про сны",
                exact_title=False,
            )
        ],
        genres=EntityFilter(include=["Science Fiction"]),
    )

    assert route_after_parse({"query_plan": plan}) == (
        "request_guess_movie_unavailable"
    )


@pytest.mark.asyncio
async def test_guess_movie_unavailable_response_is_deterministic():
    result = await request_guess_movie_unavailable({})

    assert result == {
        "final_response": (
            "Поиск фильма по описанию пока не поддерживается. "
            "Укажи точное название фильма, если оно тебе известно."
        )
    }
