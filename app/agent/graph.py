from functools import partial
from app.agent.nodes import (
    general_chat,
    # intent_classification_node,
    unified_parser_node,
    search_for_guess_node,
    search_for_recommended_node,
    synthesis_response_node,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.agent.router import route_intent
from app.agent.state import AgentState
from app.config import settings


llm_strict = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=settings.LLM_STRICT_TEMPERATURE,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_URL,
    max_retries=3
)

llm_creative = ChatOpenAI(
    model=settings.LLM_MODEL_NAME,
    temperature=settings.LLM_CREATIVE_TEMPERATURE,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_URL,
    max_retries=3
)

workflow = StateGraph(AgentState)

workflow.add_node("unified_parser", partial(unified_parser_node, llm=llm_strict))
workflow.add_node("search_for_recommended", partial(search_for_recommended_node))
workflow.add_node("search_for_guess", partial(search_for_guess_node))
workflow.add_node("general_chat", partial(general_chat, llm=llm_creative))
workflow.add_node(
    "synthesis_response", partial(synthesis_response_node, llm=llm_creative)
)

workflow.set_entry_point("unified_parser")
workflow.add_conditional_edges(
    "unified_parser",
    route_intent,
    {
        "search_for_recommended": "search_for_recommended",
        "search_for_guess": "search_for_guess",
        "general_chat": "general_chat",
    }
)

workflow.add_edge("search_for_guess", "synthesis_response")
workflow.add_edge("search_for_recommended", "synthesis_response")
workflow.add_edge("synthesis_response", END)
workflow.add_edge("general_chat", END)

graph = workflow.compile()