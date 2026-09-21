import json
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.agent.nodes import evaluate_results
from app.agent.schemas import Intent, MovieEvaluation, MovieQueryPlan
from tests.factories import make_movie_dto


@pytest.fixture
def state():
    return {
        "request_id": "evaluation-test",
        "user_id": 42,
        "user_query": "Фильмы про космос",
        "query_plan": MovieQueryPlan(intent=Intent.RECOMMEND_MOVIES),
        "candidates": [
            make_movie_dto(id=1, title="Space", overview="A journey through space."),
            make_movie_dto(id=2, title="Village", overview="Life in a village."),
            make_movie_dto(id=3, title="Moon", overview="A mission to the moon."),
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
async def test_evaluation_applies_user_facing_result_limit(state):
    state["query_plan"].result_limit = 2
    llm, _ = evaluation_llm({"accepted_movie_ids": [1, 2, 3]})

    result = await evaluate_results(state, llm)

    assert [movie.id for movie in result["selected_movies"]] == [1, 2]


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


@pytest.mark.asyncio
async def test_evaluation_limits_metadata_without_mutating_candidate(state):
    movie = make_movie_dto(
        imdb_id="tt1234567", actors=["actor"] * 312,
        directors=["director"] * 30,
        keywords=[f"keyword-{i}" for i in range(149)],
    )
    state["candidates"] = [movie]
    llm, structured = evaluation_llm(MovieEvaluation(accepted_movie_ids=[movie.id]))
    result = await evaluate_results(state, llm)
    payload = json.loads(structured.ainvoke.call_args.args[0][1].content)
    candidate = payload["candidates"][0]
    assert set(candidate) == {
        "id", "title", "original_title", "overview", "tagline", "genres",
        "release_date", "runtime", "vote_average", "keywords",
    }
    assert candidate["keywords"] == movie.keywords[:25]
    assert candidate["release_date"] == "2000-01-01"
    assert len(movie.keywords) == 149
    assert result["selected_movies"][0] is movie
