import pytest
from types import SimpleNamespace

from app.schemas import MovieDTO
from unittest.mock import AsyncMock, Mock

from app.agent.routing import route_after_parse
from app.agent.nodes import parser_query_node, general_chat, resolve_references
from app.agent.schemas import (
    Intent,
    MovieQueryPlan,
    MovieReference,
    ReferenceRelation,
    ReferenceResolutionStatus,
)



@pytest.mark.asyncio
async def test_parser_query_node():
    query_plan = MovieQueryPlan(
        intent=Intent.GENERAL_CHAT
    )

    structured_llm = Mock()
    structured_llm.ainvoke = AsyncMock(
        return_value=query_plan
    )

    llm = Mock()
    llm.with_structured_output.return_value = structured_llm

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Привет, что ты умеешь?",
    }

    result = await parser_query_node(
        state=state,
        llm=llm,
    )

    assert result == {
        "query_plan": query_plan
    }

    llm.with_structured_output.assert_called_once_with(
        MovieQueryPlan
    )

    structured_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_general_chat():
    llm = Mock()

    llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(
            content="Я могу помочь подобрать фильм."
        )
    )

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Что ты умеешь?",
    }

    result = await general_chat(
        state=state,
        llm=llm,
    )

    assert result == {
        "final_response": "Я могу помочь подобрать фильм."
    }

    llm.ainvoke.assert_awaited_once()


def test_route_general_chat():
    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Привет",
        "query_plan": MovieQueryPlan(
            intent=Intent.GENERAL_CHAT
        ),
    }

    result = route_after_parse(state)

    assert result == "general_chat"


def test_route_recommend():
    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Посоветуй фильм",
        "query_plan": MovieQueryPlan(
            intent=Intent.RECOMMEND_MOVIES
        ),
    }

    result = route_after_parse(state)

    assert result == "search_movies"


@pytest.mark.asyncio
async def test_resolve_reference_success():
    reference = MovieReference(
        query="Interstellar",
        exact_title=True,
        relation=ReferenceRelation.SIMILAR_TO,
    )

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Посоветуй что-нибудь как Interstellar",
        "query_plan": MovieQueryPlan(
            intent=Intent.RECOMMEND_MOVIES,
            reference_movies=[reference],
        ),
    }

    movie = MovieDTO(
        id=1,
        title="Interstellar",
        overview="A team travels through a wormhole in space.",
        genres="Science Fiction, Drama",
        credits="Christopher Nolan",
        tagline="Mankind was born on Earth.",
        release_date="2014-11-07",
        vote_average=8.7,
        keywords="space, wormhole, future",
    )

    movie_search = Mock()
    movie_search.resolve_reference = AsyncMock(
        return_value=[movie]
    )

    result = await resolve_references(
        state=state,
        movie_search=movie_search,
    )

    resolved = result["resolved_references"]

    assert len(resolved) == 1

    assert resolved[0].reference == reference
    assert resolved[0].movie == movie
    assert resolved[0].status == ReferenceResolutionStatus.RESOLVED

    movie_search.resolve_reference.assert_awaited_once_with(
        query="Interstellar",
        exact_title=True,
        year=None,
    )


@pytest.mark.asyncio
async def test_resolve_reference_not_found():
    reference = MovieReference(
        query="Unknown Movie",
        exact_title=True,
        relation=ReferenceRelation.SIMILAR_TO,
    )

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Посоветуй что-нибудь как Unknown Movie",
        "query_plan": MovieQueryPlan(
            intent=Intent.RECOMMEND_MOVIES,
            reference_movies=[reference],
        ),
    }

    movie_search = Mock()
    movie_search.resolve_reference = AsyncMock(
        return_value=[]
    )

    result = await resolve_references(
        state=state,
        movie_search=movie_search,
    )

    resolved = result["resolved_references"]

    assert len(resolved) == 1

    assert resolved[0].reference == reference
    assert resolved[0].movie is None
    assert resolved[0].status == ReferenceResolutionStatus.NOT_FOUND

    movie_search.resolve_reference.assert_awaited_once_with(
        query="Unknown Movie",
        exact_title=True,
        year=None,
    )


@pytest.mark.asyncio
async def test_resolve_multiple_references():
    interstellar_reference = MovieReference(
        query="Interstellar",
        exact_title=True,
        relation=ReferenceRelation.SIMILAR_TO,
    )

    gravity_reference = MovieReference(
        query="Gravity",
        exact_title=True,
        relation=ReferenceRelation.EXCLUDE,
    )

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": (
            "Посоветуй что-нибудь как Interstellar, "
            "но не как Gravity"
        ),
        "query_plan": MovieQueryPlan(
            intent=Intent.RECOMMEND_MOVIES,
            reference_movies=[
                interstellar_reference,
                gravity_reference,
            ],
        ),
    }

    interstellar = MovieDTO(
        id=1,
        title="Interstellar",
        release_date="2014-11-07",
        vote_average=8.7,
    )

    gravity = MovieDTO(
        id=2,
        title="Gravity",
        release_date="2013-10-04",
        vote_average=7.7,
    )

    movie_search = Mock()
    movie_search.resolve_reference = AsyncMock(
        side_effect=[
            [interstellar],
            [gravity],
        ]
    )

    result = await resolve_references(
        state=state,
        movie_search=movie_search,
    )

    resolved = result["resolved_references"]

    assert len(resolved) == 2

    assert resolved[0].reference == interstellar_reference
    assert resolved[0].movie == interstellar
    assert resolved[0].status == ReferenceResolutionStatus.RESOLVED
    assert (
        resolved[0].reference.relation
        == ReferenceRelation.SIMILAR_TO
    )

    assert resolved[1].reference == gravity_reference
    assert resolved[1].movie == gravity
    assert resolved[1].status == ReferenceResolutionStatus.RESOLVED
    assert (
        resolved[1].reference.relation
        == ReferenceRelation.EXCLUDE
    )

    assert movie_search.resolve_reference.await_count == 2

    movie_search.resolve_reference.assert_any_await(
        query="Interstellar",
        exact_title=True,
        year=None,
    )

    movie_search.resolve_reference.assert_any_await(
        query="Gravity",
        exact_title=True,
        year=None,
    )


@pytest.mark.asyncio
async def test_resolve_multiple_references_with_not_found():
    first_reference = MovieReference(
        query="Interstellar",
        exact_title=True,
    )

    second_reference = MovieReference(
        query="Unknown Movie",
        exact_title=True,
    )

    state = {
        "request_id": "test-request",
        "user_id": 1,
        "user_query": "Как Interstellar и Unknown Movie",
        "query_plan": MovieQueryPlan(
            intent=Intent.RECOMMEND_MOVIES,
            reference_movies=[
                first_reference,
                second_reference,
            ],
        ),
    }

    interstellar = MovieDTO(
        id=1,
        title="Interstellar",
    )

    movie_search = Mock()
    movie_search.resolve_reference = AsyncMock(
        side_effect=[
            [interstellar],
            [],
        ]
    )

    result = await resolve_references(
        state=state,
        movie_search=movie_search,
    )

    resolved = result["resolved_references"]

    assert len(resolved) == 2

    assert resolved[0].movie == interstellar
    assert resolved[0].status == ReferenceResolutionStatus.RESOLVED

    assert resolved[1].movie is None
    assert resolved[1].status == ReferenceResolutionStatus.NOT_FOUND
