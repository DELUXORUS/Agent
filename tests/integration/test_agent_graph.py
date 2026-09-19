import runpy
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.schemas import (
    FloatRange,
    Intent,
    IntRange,
    MovieEvaluation,
    MovieQueryPlan,
)


@pytest.mark.asyncio
async def test_graph_uses_real_service_and_postgresql(
    monkeypatch,
    pg_catalog: AsyncSession,
    query_embedding: list[float],
):
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        semantic_query="space",
        year=IntRange(min=2010, max=2020),
        rating=FloatRange(min=7.0, max=8.0),
        limit=3,
    )
    parser = Mock(ainvoke=AsyncMock(return_value=plan))
    evaluator = Mock(ainvoke=AsyncMock(return_value=MovieEvaluation(
        accepted_movie_ids=[3],
    )))
    llm = Mock()

    def structured_output(schema):
        if schema is MovieQueryPlan:
            return parser
        if schema is MovieEvaluation:
            return evaluator
        raise AssertionError(f"Unexpected output schema: {schema}")

    llm.with_structured_output.side_effect = structured_output
    llm.ainvoke = AsyncMock(side_effect=AssertionError(
        "Unexpected unstructured LLM call"
    ))
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=llm))

    embedder = Mock(
        get_embedding=AsyncMock(return_value=query_embedding)
    )
    embedder_module = ModuleType("app.services.embedder")
    embedder_module.embedder = embedder
    monkeypatch.setitem(sys.modules, "app.services.embedder", embedder_module)

    @asynccontextmanager
    async def session_factory():
        yield pg_catalog

    database_module = ModuleType("app.db.database")
    database_module.async_session_maker = session_factory
    monkeypatch.setitem(sys.modules, "app.db.database", database_module)

    graph_path = Path(__file__).resolve().parents[2] / "app/agent/graph.py"
    graph = runpy.run_path(str(graph_path))["graph"]

    result = await graph.ainvoke({
        "request_id": "postgres-graph-test",
        "user_id": 42,
        "user_query": "Посоветуй космический фильм с рейтингом от 7 до 8",
    })

    # Movie 1 satisfies the hard filters but user 42 has already watched it.
    assert [movie.id for movie in result["candidates"]] == [2, 3]
    assert [movie.id for movie in result["selected_movies"]] == [3]
    assert "Moon station" in result["final_response"]
    assert "Orbit" not in result["final_response"]
    embedder.get_embedding.assert_awaited_once_with("space")
    parser.ainvoke.assert_awaited_once()
    evaluator.ainvoke.assert_awaited_once()
