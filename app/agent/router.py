import logging
from app.agent.state import AgentState

logger = logging.getLogger("uvicorn")

def route_intent(state: AgentState):
    intent = state.get("intent")

    logger.info(f"ROUTER: Перенаправление по интенту -> '{intent}'")


    if intent == "guess_movie":
        return "parse_guess_filter"

    elif intent == "recommend_movies":
        return "parse_recommend_filter"

    return "general_chat"