from app.agent.schemas import Intent
from app.agent.state import AgentState
from app.agent.schemas import ReferenceResolutionStatus


def route_after_parse(state: AgentState) -> str:
    query_plan = state.get("query_plan")

    if query_plan is None:
        raise RuntimeError(
            "query_plan is missing after parse_querynode"
        )

    if query_plan.intent == Intent.GENERAL_CHAT:
        return "general_chat"

    if query_plan.intent == Intent.GUESS_MOVIE:
        return "request_guess_movie_unavailable"

    if query_plan.reference_movies:
        return "resolve_references"

    return "search_movies"


def route_after_resolve_references(state: AgentState) -> str:
    resolved_references = state["resolved_references"]

    for resolved_reference in resolved_references:
        if resolved_reference.status != ReferenceResolutionStatus.RESOLVED:
            return "request_reference_clarification"

    return "search_movies"
