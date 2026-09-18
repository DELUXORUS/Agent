import json
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.agent.nodes import evaluate_results
from app.agent.schemas import MovieEvaluation
from app.schemas import MovieDTO


@pytest.fixture
def state():
    return {
        "request_id": "evaluation-test",
        "user_id": 42,
        "user_query": "Фильмы про космос",
        "candidates": [
            MovieDTO(id=1, title="Space", overview="A journey through space."),
            MovieDTO(id=2, title="Village", overview="Life in a village."),
            MovieDTO(id=3, title="Moon", overview="A mission to the moon."),
        ],
    }


def evaluation_llm(response):
    structured = Mock(ainvoke=AsyncMock(return_value=response))
    llm = Mock()
    llm.with_structured_output.return_value = structured
    return llm, structured


@pytest.mark.asyncio
async def test_evaluation_keeps_original_movies_and_search_order(state):
    llm, structured = evaluation_llm(MovieEvaluation(accepted_movie_ids=[3, 1]))
    result = await evaluate_results(state, llm)
    assert [movie.id for movie in result["selected_movies"]] == [1, 3]
    assert result["selected_movies"][0] is state["candidates"][0]
    assert len(state["candidates"]) == 3
    payload = json.loads(structured.ainvoke.call_args.args[0][1].content)
    assert payload["user_query"] == state["user_query"]
    assert payload["candidates"][0]["overview"] == state["candidates"][0].overview


@pytest.mark.asyncio
async def test_empty_candidates_skip_llm(state):
    state["candidates"] = []
    llm = Mock()
    assert await evaluate_results(state, llm) == {"selected_movies": []}
    llm.with_structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_all_rejected_is_a_successful_empty_selection(state):
    llm, _ = evaluation_llm({"accepted_movie_ids": []})
    assert await evaluate_results(state, llm) == {"selected_movies": []}


@pytest.mark.asyncio
async def test_unknown_id_rejects_whole_evaluation(state):
    llm, _ = evaluation_llm({"accepted_movie_ids": [1, 999]})
    with pytest.raises(ValueError, match="outside the candidate list"):
        await evaluate_results(state, llm)


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [
    {}, None,
    {"accepted_movie_ids": ["1"]}, {"accepted_movie_ids": [True]},
    {"accepted_movie_ids": [], "extra": "unexpected"},
])
async def test_invalid_output_is_not_treated_as_no_matches(state, response):
    llm, _ = evaluation_llm(response)
    with pytest.raises(ValidationError):
        await evaluate_results(state, llm)


@pytest.mark.asyncio
async def test_llm_failure_propagates_without_fallback(state):
    llm, structured = evaluation_llm(None)
    structured.ainvoke.side_effect = TimeoutError("LLM timed out")
    with pytest.raises(TimeoutError):
        await evaluate_results(state, llm)
    assert "selected_movies" not in state


@pytest.mark.parametrize("ids,expected", [
    ([3, 1, 3, 2, 1], [3, 1, 2]),
    ([1, 1], [1]),
    ([2, 1], [2, 1]),
    ([], []),
])
def test_evaluation_schema_removes_duplicates_preserving_order(ids, expected):
    evaluation = MovieEvaluation(accepted_movie_ids=ids)
    assert evaluation.accepted_movie_ids == expected
