import pytest

from app.agent.capabilities import get_unsupported_filters
from app.agent.nodes import request_filter_adjustment
from app.agent.routing import route_after_parse
from app.agent.schemas import EntityFilter, Intent, IntRange, MovieQueryPlan


@pytest.mark.parametrize(
    ("plan_changes", "expected_filter"),
    [
        ({"runtime_minutes": IntRange(min=90)}, "runtime"),
        ({"genres": EntityFilter(include=["Drama"])}, "genres"),
        ({"genres": EntityFilter(exclude=["Horror"])}, "genres"),
        ({"actors": EntityFilter(include=["Actor"])}, "actors"),
        ({"actors": EntityFilter(exclude=["Actor"])}, "actors"),
        ({"directors": EntityFilter(include=["Director"])}, "directors"),
        ({"directors": EntityFilter(exclude=["Director"])}, "directors"),
    ],
)
def test_detects_each_unsupported_filter(plan_changes, expected_filter):
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        **plan_changes,
    )

    assert get_unsupported_filters(plan) == [expected_filter]
    assert route_after_parse({"query_plan": plan}) == "request_filter_adjustment"


def test_supported_filters_continue_to_search():
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        year=IntRange(min=2000, max=2020),
    )

    assert get_unsupported_filters(plan) == []
    assert route_after_parse({"query_plan": plan}) == "search_movies"


def test_general_chat_has_priority_over_filter_capabilities():
    plan = MovieQueryPlan(
        intent=Intent.GENERAL_CHAT,
        genres=EntityFilter(include=["Drama"]),
    )

    assert route_after_parse({"query_plan": plan}) == "general_chat"


@pytest.mark.asyncio
async def test_filter_adjustment_explains_all_unsupported_filters():
    plan = MovieQueryPlan(
        intent=Intent.RECOMMEND_MOVIES,
        runtime_minutes=IntRange(max=120),
        actors=EntityFilter(include=["Actor"]),
        directors=EntityFilter(exclude=["Director"]),
    )

    result = await request_filter_adjustment({"query_plan": plan})

    assert result["unsupported_filters"] == [
        "runtime",
        "actors",
        "directors",
    ]
    assert "хронометражу, актёрам и режиссёрам" in result["final_response"]
    assert "году и рейтингу" in result["final_response"]


@pytest.mark.asyncio
async def test_filter_adjustment_requires_unsupported_filter():
    plan = MovieQueryPlan(intent=Intent.RECOMMEND_MOVIES)

    with pytest.raises(ValueError, match="unsupported filter"):
        await request_filter_adjustment({"query_plan": plan})
