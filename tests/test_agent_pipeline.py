"""Run the production graph and real nodes with mocked external dependencies."""
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent.schemas import (
    FloatRange, Intent, IntRange, MovieEvaluation, MovieQueryPlan,
    MovieReference, ReferenceRelation, ReferenceResolutionStatus,
)
from app.schemas import MovieDTO
from app.services.schemas import MovieSearchParams


@pytest.fixture
def pipeline(monkeypatch):
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        semantic_query="space exploration",
        year=IntRange(min=2000, max=2024),
        rating=FloatRange(min=7),
        limit=3,
    )
    movies = [
        MovieDTO(id=20, title="Interstellar", release_date="2014-11-07",
                 vote_average=8.4, overview="Explorers travel through a wormhole."),
        MovieDTO(id=10, title="Village", release_date="2010-01-01",
                 vote_average=7.5, overview="Life in a village."),
        MovieDTO(id=30, title="Moon", release_date="2009-06-12",
                 vote_average=7.6, overview="A mission to the moon."),
    ]
    parser = Mock(ainvoke=AsyncMock(return_value=plan))
    evaluator = Mock(ainvoke=AsyncMock(return_value=MovieEvaluation(
        accepted_movie_ids=[30, 20],
    )))
    llm = Mock()

    def structured_output(schema):
        if schema is MovieQueryPlan:
            return parser
        if schema is MovieEvaluation:
            return evaluator
        raise AssertionError(f"Unexpected output schema: {schema}")

    llm.with_structured_output.side_effect = structured_output
    llm.ainvoke = AsyncMock(side_effect=AssertionError("Unexpected unstructured LLM call"))
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=llm))
    service = Mock(
        search_recommendations=AsyncMock(return_value=movies),
        resolve_reference=AsyncMock(side_effect=AssertionError("Unexpected reference lookup")),
    )
    monkeypatch.setattr("app.services.movie_search.MovieSearchService", Mock(return_value=service))

    # graph.py imports global resources; replace those modules before loading it.
    # Nodes, routing, mappers and the graph definition are left unchanged.
    embedder_module = ModuleType("app.services.embedder")
    embedder_module.embedder = Mock()
    database_module = ModuleType("app.db.database")
    database_module.async_session_maker = Mock()
    monkeypatch.setitem(sys.modules, "app.services.embedder", embedder_module)
    monkeypatch.setitem(sys.modules, "app.db.database", database_module)
    path = Path(__file__).resolve().parents[1] / "app/agent/graph.py"
    graph = runpy.run_path(str(path))["graph"]
    return SimpleNamespace(
        graph=graph, parser=parser, evaluator=evaluator, llm=llm,
        service=service, movies=movies, plan=plan,
        initial={"request_id": "pipeline-test", "user_id": 42,
                 "user_query": "Посоветуй 3 фильма про космос за 2000–2024 годы с рейтингом от 7"},
    )


@pytest.mark.asyncio
async def test_recommendation_graph_produces_filtered_final_response(pipeline):
    result = await pipeline.graph.ainvoke(pipeline.initial)

    assert [movie.id for movie in result["candidates"]] == [20, 10, 30]
    assert [movie.id for movie in result["selected_movies"]] == [20, 30]
    text = result["final_response"]
    assert "1. Interstellar — 2014" in text
    assert "2. Moon — 2009" in text
    assert "Village" not in text
    pipeline.service.search_recommendations.assert_awaited_once_with(
        42, MovieSearchParams(semantic_query="space exploration", year_min=2000,
                              year_max=2024, rating_min=7, limit=3),
    )
    pipeline.service.resolve_reference.assert_not_awaited()
    pipeline.parser.ainvoke.assert_awaited_once()
    pipeline.evaluator.ainvoke.assert_awaited_once()
    assert pipeline.parser.ainvoke.call_args.args[0][1].content == pipeline.initial["user_query"]
    payload = json.loads(pipeline.evaluator.ainvoke.call_args.args[0][1].content)
    assert payload == {
        "user_query": pipeline.initial["user_query"],
        "candidates": [movie.model_dump(mode="json") for movie in pipeline.movies],
    }
    pipeline.llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_search_skips_evaluation_and_returns_no_matches(pipeline):
    pipeline.service.search_recommendations.return_value = []
    result = await pipeline.graph.ainvoke(pipeline.initial)

    assert result["candidates"] == []
    assert result["selected_movies"] == []
    assert "не удалось найти подходящие фильмы" in result["final_response"]
    pipeline.service.search_recommendations.assert_awaited_once()
    pipeline.evaluator.ainvoke.assert_not_awaited()
    pipeline.llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluator_rejecting_everything_returns_no_matches(pipeline):
    pipeline.evaluator.ainvoke.return_value = MovieEvaluation(accepted_movie_ids=[])
    result = await pipeline.graph.ainvoke(pipeline.initial)

    assert len(result["candidates"]) == 3
    assert result["selected_movies"] == []
    assert "не удалось найти подходящие фильмы" in result["final_response"]
    assert all(movie.title not in result["final_response"] for movie in pipeline.movies)
    pipeline.evaluator.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["parser", "evaluator"])
async def test_llm_failure_stops_graph_without_final_response(pipeline, failed_stage):
    failure = TimeoutError(f"{failed_stage} timed out")
    getattr(pipeline, failed_stage).ainvoke.side_effect = failure
    updates = []

    with pytest.raises(TimeoutError) as caught:
        async for update in pipeline.graph.astream(pipeline.initial, stream_mode="updates"):
            updates.append(update)

    assert caught.value is failure
    assert all("compose_response" not in update for update in updates)
    if failed_stage == "parser":
        pipeline.service.search_recommendations.assert_not_awaited()
        pipeline.evaluator.ainvoke.assert_not_awaited()
    else:
        pipeline.service.search_recommendations.assert_awaited_once()
        pipeline.evaluator.ainvoke.assert_awaited_once()
        assert any("search_movies" in update for update in updates)


@pytest.mark.asyncio
async def test_unknown_evaluated_id_stops_graph_before_composition(pipeline):
    pipeline.evaluator.ainvoke.return_value = MovieEvaluation(accepted_movie_ids=[20, 999])
    updates = []
    with pytest.raises(ValueError, match="outside the candidate list"):
        async for update in pipeline.graph.astream(pipeline.initial, stream_mode="updates"):
            updates.append(update)
    assert any("search_movies" in update for update in updates)
    assert all("compose_response" not in update for update in updates)


@pytest.mark.asyncio
async def test_resolved_references_continue_to_recommendations(pipeline):
    pipeline.plan.reference_movies = [
        MovieReference(query="Dune", exact_title=True, year=1984),
        MovieReference(query="Gravity", exact_title=True, relation=ReferenceRelation.EXCLUDE),
    ]
    pipeline.service.resolve_reference.side_effect = [
        [MovieDTO(id=100, title="Dune", release_date="1984-12-14")],
        [MovieDTO(id=200, title="Gravity", release_date="2013-10-04")],
    ]
    result = await pipeline.graph.ainvoke(pipeline.initial)

    assert all(item.status == ReferenceResolutionStatus.RESOLVED
               for item in result["resolved_references"])
    pipeline.service.resolve_reference.assert_any_await(query="Dune", exact_title=True, year=1984)
    pipeline.service.resolve_reference.assert_any_await(query="Gravity", exact_title=True, year=None)
    pipeline.service.search_recommendations.assert_awaited_once_with(
        42, MovieSearchParams(semantic_query="space exploration", year_min=2000,
                              year_max=2024, rating_min=7, limit=3,
                              similar_movie_ids=[100], excluded_movie_ids=[200]),
    )
    assert "1. Interstellar" in result["final_response"]
    assert "Повтори полный запрос" not in result["final_response"]
