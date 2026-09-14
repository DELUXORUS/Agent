from app.agent.nodes import (
    parser_query_node, resolve_references,
    search_movies, evaluate_results,
    compose_response, movie_request,
    general_chat,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.agent.routing import route_intent, route_references
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

workflow.add_node("parser_query_node", parser_query_node)
workflow.add_node("general_chat", general_chat)
workflow.add_node("movie_request", movie_request)
workflow.add_node("resolve_references", resolve_references)
workflow.add_node("search_movies", search_movies)
workflow.add_node("evaluate_results", evaluate_results)
workflow.add_node("compose_response", compose_response)

workflow.set_entry_point("parser_query_node")
workflow.add_conditional_edges(
    "parser_query_node",
    route_intent,
    {
        "recommend": "movie_request",
        "guess_movie": "movie_request",
        "chat": "general_chat",
    }
)
workflow.add_edge("general_chat", END)
workflow.add_conditional_edges(
    "movie_request",
    route_references,
    {
        "resolve_references": "resolve_references",
        "search_movies": "search_movies",
    }
)
workflow.add_edge("resolve_references", "search_movies")
workflow.add_edge("search_movies", "evaluate_results")
workflow.add_edge("evaluate_results", "compose_response")
workflow.add_edge("compose_response", END)


# workflow.add_node("unified_parser", partial(unified_parser_node, llm=llm_strict))
# workflow.add_node("search_for_recommended", partial(search_for_recommended_node))
# workflow.add_node("search_for_guess", partial(search_for_guess_node))
# workflow.add_node("general_chat", partial(general_chat, llm=llm_creative))
# workflow.add_node(
#     "synthesis_response", partial(synthesis_response_node, llm=llm_creative)
# )
#
# workflow.set_entry_point("unified_parser")
# workflow.add_conditional_edges(
#     "unified_parser",
#     route_intent,
#     {
#         "recommend": "search_movies",
#         "guess_movie": "search_movies",
#         "chat": "general_chat",
#     }
# )
#
# workflow.add_edge("search_for_guess", "synthesis_response")
# workflow.add_edge("search_for_recommended", "synthesis_response")
# workflow.add_edge("synthesis_response", END)
# workflow.add_edge("general_chat", END)

graph = workflow.compile()