from app.agent.nodes import (
    parser_query_node, resolve_references,
    search_movies, evaluate_results,
    compose_response,
    general_chat,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from functools import partial
from app.agent.routing import route_after_parse
from app.agent.state import AgentState
from app.config import settings

from app.services.movie_search import MovieSearchService
from app.services.embedder import embedder
from app.db.database import async_session_maker


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


movie_search_service = MovieSearchService(
    session_factory=async_session_maker,
    embedder=embedder,
)


workflow = StateGraph(AgentState)

workflow.add_node(
    "parser_query_node",
    partial(
        parser_query_node,
        llm=llm_creative
    )
)

workflow.add_node(
    "general_chat",
    partial(
        general_chat,
        llm=llm_creative
    )
)

workflow.add_node(
    "resolve_references",
    partial(
        resolve_references,
        movie_search=movie_search_service
    )
)

workflow.add_node(
    "search_movies",
    partial(
      search_movies,
      movie_search=movie_search_service
    )
)
workflow.add_node("evaluate_results", evaluate_results)
workflow.add_node("compose_response", compose_response)

workflow.set_entry_point("parser_query_node")
workflow.add_conditional_edges(
    "parser_query_node",
    route_after_parse,
    {
        "resolve_references": "resolve_references",
        "search_movies": "search_movies",
        "general_chat": "general_chat",
    }
)
workflow.add_edge("resolve_references", "search_movies")
workflow.add_edge("search_movies", "evaluate_results")
workflow.add_edge("evaluate_results", "compose_response")
workflow.add_edge("compose_response", END)

graph = workflow.compile()