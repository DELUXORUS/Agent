from app.agent.schemas import Intent
from app.agent.state import AgentState
from app.agent.schemas import ReferenceResolutionStatus
from app.agent.capabilities import get_unsupported_filters


def route_after_parse(state: AgentState) -> str:
    query_plan = state.get("query_plan")

    if query_plan is None:
        raise RuntimeError(
            "query_plan is missing after parse_querynode"
        )

    if query_plan.intent == Intent.GENERAL_CHAT:
        return "general_chat"

    if get_unsupported_filters(query_plan):
        return "request_filter_adjustment"

    if query_plan.reference_movies:
        return "resolve_references"

    return "search_movies"


def route_after_resolve_references(state: AgentState) -> str:
    resolved_references = state["resolved_references"]

    for resolved_reference in resolved_references:
        if resolved_reference.status == ReferenceResolutionStatus.AMBIGUOUS \
            or resolved_reference.status == ReferenceResolutionStatus.NOT_FOUND:
            return "request_reference_clarification"

    return "search_movies"
