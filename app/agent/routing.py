import logging
from app.agent.state import AgentState
from schemas import Intent

logger = logging.getLogger("uvicorn")

def route_intent(state: AgentState):
    query_plan = state.get("query_plan")

    if query_plan is None:
        raise RuntimeError(
            "query_plan is missing after parse_query node"
        )
    logger.info(f"ROUTER: Перенаправление по интенту -> '{query_plan.intent}'")

    if query_plan.intent == Intent.RECOMMEND:
        return "recommend"

    if query_plan.intent == Intent.GUESS_MOVIE:
        return "guess_movie"

    return "chat"