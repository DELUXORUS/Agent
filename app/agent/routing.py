from app.agent.schemas import Intent
from app.agent.state import AgentState


def route_after_parse(state: AgentState) -> str:
    query_plan = state.get("query_plan")

    if query_plan is None:
        raise RuntimeError(
            "query_plan is missing after parse_querynode"
        )

    if query_plan.intent == Intent.GENERAL_CHAT:
        return "general_chat"

    if query_plan.reference_movies:
        return "resolve_references"

    return "search_movies"