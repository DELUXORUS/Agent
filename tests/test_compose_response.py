from unittest.mock import AsyncMock, Mock

import pytest

from app.agent.nodes import compose_response, evaluate_results
from app.agent.schemas import MovieEvaluation
from tests.factories import make_movie_dto


@pytest.mark.asyncio
async def test_empty_selection_returns_no_matches_without_calling_llm():
    llm = Mock()
    result = await compose_response({"selected_movies": []}, llm)

    assert result == {
        "final_response": "В нашем каталоге не удалось найти подходящие фильмы."
    }
    assert llm.mock_calls == []


@pytest.mark.asyncio
async def test_response_preserves_selection_order_and_does_not_include_rejected_movies():
    first = make_movie_dto(id=20, title="Moon", release_date="2009-06-12", vote_average=7.6)
    second = make_movie_dto(id=10, title="Interstellar", release_date="2014-11-07", vote_average=8.4)
    rejected = make_movie_dto(id=30, title="Rejected movie")
    state = {
        "candidates": [second, rejected, first],
        "selected_movies": [first, second],
    }
    llm = Mock()

    result = await compose_response(state, llm)

    assert result["final_response"] == (
        "Вот подходящие фильмы:\n\n"
        "1. Moon — 2009 · рейтинг: 7.6\n"
        "2. Interstellar — 2014 · рейтинг: 8.4"
    )
    assert state["selected_movies"] == [first, second]
    assert "Rejected movie" not in result["final_response"]
    assert llm.mock_calls == []


@pytest.mark.asyncio
async def test_movie_titles_are_escaped_for_telegram_html():
    movie = make_movie_dto(id=1, title='<b>Tom & "Jerry"</b>')

    result = await compose_response({"selected_movies": [movie]}, Mock())

    assert "&lt;b&gt;Tom &amp; &quot;Jerry&quot;&lt;/b&gt;" in result["final_response"]
    assert "<b>" not in result["final_response"]
    assert movie.title == '<b>Tom & "Jerry"</b>'


@pytest.mark.asyncio
@pytest.mark.parametrize("release_date,rating,expected", [
    ("2020-01-01", 0.0, "2020 · рейтинг: 0.0"),
])
async def test_date_and_zero_rating(release_date, rating, expected):
    movie = make_movie_dto(id=1, title="Example", release_date=release_date, vote_average=rating)

    result = await compose_response({"selected_movies": [movie]}, Mock())

    assert expected in result["final_response"]
    assert "None" not in result["final_response"]


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_ids", [[2], []])
async def test_evaluation_result_is_used_by_response_node(accepted_ids):
    candidates = [
        make_movie_dto(id=1, title="Village", overview="Life in a village."),
        make_movie_dto(id=2, title="Moon", overview="A mission to the moon."),
    ]
    state = {"user_query": "Movies about space", "candidates": candidates}
    structured = Mock(ainvoke=AsyncMock(return_value=MovieEvaluation(
        accepted_movie_ids=accepted_ids,
    )))
    evaluator = Mock()
    evaluator.with_structured_output.return_value = structured

    state.update(await evaluate_results(state, evaluator))
    composer_llm = Mock()
    result = await compose_response(state, composer_llm)

    assert "Village" not in result["final_response"]
    if accepted_ids:
        assert "1. Moon" in result["final_response"]
    else:
        assert "не удалось найти подходящие фильмы" in result["final_response"]
        assert "Moon" not in result["final_response"]
    structured.ainvoke.assert_awaited_once()
    assert composer_llm.mock_calls == []
