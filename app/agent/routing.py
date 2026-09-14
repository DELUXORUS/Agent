from schema import Intent
from app.agent.state import AgentState


def route_intent(state: AgentState) -> str:
    query_plan = state.get("query_plan")

    if query_plan is None:
        raise RuntimeError(
            "query_plan is missing after parse_query node"
        )

    if query_plan.intent == Intent.RECOMMEND_MOVIES:
        return "recommend"

    if query_plan.intent == Intent.GUESS_MOVIE:
        return "guess_movie"

    return "chat"


def route_references(state: AgentState):
    query_plan = state.get("query_plan")

    if not query_plan.references_movie:
        return "search_movies"
    else:
        return "resolve_references"