import logging
from app.agent.state import AgentState

logger = logging.getLogger("uvicorn")

def route_intent(state: AgentState):
    intent = state.get("intent")

    logger.info(f"ROUTER: Перенаправление по интенту -> '{intent}'")

    if intent == "recommend_movies":
        return "search_for_recommended"
    elif intent == "guess_movie":
        return "search_for_guess"

    return "general_chat"