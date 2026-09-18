"""Clarification behavior, including routing through the actual application graph.

External dependencies are replaced; no LLM, embedder or database is started.
"""
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest

from app.agent.nodes import request_reference_clarification
from app.agent.routing import route_after_resolve_references
from app.agent.schemas import (
    Intent,
    MovieQueryPlan,
    MovieReference,
    ReferenceResolutionStatus as Status,
    ResolvedMovieReference,
)
from app.schemas import MovieDTO


def reference_result(status, query="Dune"):
    movies = [
        MovieDTO(id=1, title=query, release_date="1984-12-14"),
        MovieDTO(id=2, title=query, release_date="2021-10-22"),
    ]
    return ResolvedMovieReference(
        reference=MovieReference(query=query, exact_title=True),
        status=status,
        movie=movies[0] if status == Status.RESOLVED else None,
        candidates=movies if status == Status.AMBIGUOUS else [],
    )


@pytest.mark.parametrize("statuses,expected", [
    ([Status.RESOLVED, Status.AMBIGUOUS], "request_reference_clarification"),
    ([Status.AMBIGUOUS, Status.RESOLVED], "request_reference_clarification"),
    ([Status.RESOLVED, Status.NOT_FOUND], "request_reference_clarification"),
    ([Status.NOT_FOUND, Status.RESOLVED], "request_reference_clarification"),
    ([Status.NOT_FOUND, Status.AMBIGUOUS], "request_reference_clarification"),
    ([Status.AMBIGUOUS, Status.NOT_FOUND], "request_reference_clarification"),
    ([Status.RESOLVED, Status.RESOLVED], "search_movies"),
    ([], "search_movies"),
])
def test_routing_checks_all_references(statuses, expected):
    state = {"resolved_references": [reference_result(status) for status in statuses]}
    assert route_after_resolve_references(state) == expected


@pytest.mark.asyncio
async def test_clarification_lists_candidates_and_requests_full_query():
    state = {"resolved_references": [reference_result(Status.AMBIGUOUS)]}
    result = await request_reference_clarification(state)
    text = result["final_response"]
    assert "Dune — 1984" in text
    assert "Dune — 2021" in text
    assert text.index("1984") < text.index("2021")
    assert "Повтори полный запрос" in text
    assert "selected_movies" not in result


@pytest.mark.asyncio
async def test_missing_movie_gets_not_found_message():
    result = await request_reference_clarification({
        "resolved_references": [reference_result(Status.NOT_FOUND, "Missing")],
    })
    assert "Не удалось найти в каталоге фильм «Missing»" in result["final_response"]
    assert "Проверь название" in result["final_response"]
    assert "найдено несколько вариантов" not in result["final_response"]


@pytest.mark.asyncio
async def test_multiple_problems_are_kept_and_resolved_movies_are_omitted():
    state = {"resolved_references": [
        reference_result(Status.AMBIGUOUS, "Dune"),
        reference_result(Status.RESOLVED, "Interstellar"),
        reference_result(Status.NOT_FOUND, "Missing"),
        reference_result(Status.AMBIGUOUS, "Another movie"),
    ]}
    result = await request_reference_clarification(state)
    text = result["final_response"]
    for title in ("Dune", "Missing", "Another movie"):
        assert title in text
    assert "Interstellar" not in text
    assert text.count("Повтори полный запрос") == 1


@pytest.mark.asyncio
async def test_clarification_escapes_html_and_handles_unknown_dates():
    ambiguous = reference_result(Status.AMBIGUOUS, '<b>Dune & "friends"</b>')
    ambiguous.candidates[0].release_date = None
    missing = reference_result(Status.NOT_FOUND, "<unknown>&")
    result = await request_reference_clarification({"resolved_references": [ambiguous, missing]})
    text = result["final_response"]
    assert "год неизвестен" in text
    assert "&lt;b&gt;Dune &amp; &quot;friends&quot;&lt;/b&gt;" in text
    assert "&lt;unknown&gt;&amp;" in text
    assert "<b>" not in text and "<unknown>" not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [[], [Status.RESOLVED]])
async def test_clarification_rejects_call_without_unresolved_references(statuses):
    with pytest.raises(ValueError, match="unresolved reference"):
        await request_reference_clarification({
            "resolved_references": [reference_result(status) for status in statuses],
        })


@pytest.mark.asyncio
@pytest.mark.parametrize("statuses", [
    [Status.RESOLVED, Status.AMBIGUOUS],
    [Status.AMBIGUOUS, Status.RESOLVED],
    [Status.RESOLVED, Status.NOT_FOUND],
    [Status.NOT_FOUND, Status.RESOLVED],
    [Status.AMBIGUOUS, Status.NOT_FOUND],
    [Status.AMBIGUOUS, Status.AMBIGUOUS],
])
async def test_actual_graph_ends_after_clarification(monkeypatch, statuses):
    references = [reference_result(status, f"Movie {idx}") for idx, status in enumerate(statuses)]
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        reference_movies=[item.reference for item in references],
    )
    structured = Mock(ainvoke=AsyncMock(return_value=plan))
    llm = Mock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=llm))

    service = Mock()
    service.resolve_reference = AsyncMock(side_effect=[
        [item.movie] if item.status == Status.RESOLVED else item.candidates
        for item in references
    ])
    service.search_recommendations = AsyncMock(side_effect=AssertionError("Search must not run"))
    monkeypatch.setattr("app.services.movie_search.MovieSearchService", Mock(return_value=service))

    # Prevent graph.py's import-time creation of the real embedder and DB engine.
    embedder_module = ModuleType("app.services.embedder")
    embedder_module.embedder = Mock()
    database_module = ModuleType("app.db.database")
    database_module.async_session_maker = Mock()
    monkeypatch.setitem(sys.modules, "app.services.embedder", embedder_module)
    monkeypatch.setitem(sys.modules, "app.db.database", database_module)

    downstream_calls = []

    async def unexpected_downstream(state, llm):
        downstream_calls.append(True)
        raise AssertionError("Evaluation and composition must not run during clarification")

    monkeypatch.setattr("app.agent.nodes.evaluate_results", unexpected_downstream)
    monkeypatch.setattr("app.agent.nodes.compose_response", unexpected_downstream)
    graph_path = Path(__file__).resolve().parents[1] / "app/agent/graph.py"
    graph = runpy.run_path(str(graph_path))["graph"]
    result = await graph.ainvoke({
        "request_id": "clarification-test", "user_id": 42, "user_query": "Find similar movies",
    })
    assert "Повтори полный запрос" in result["final_response"]
    assert [item.status for item in result["resolved_references"]] == statuses
    assert "candidates" not in result
    assert service.resolve_reference.await_count == len(references)
    service.search_recommendations.assert_not_awaited()
    assert downstream_calls == []
