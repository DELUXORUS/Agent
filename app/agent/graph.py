from app.agent.nodes import (
    parser_query_node, resolve_references,
    search_movies, evaluate_results,
    compose_response,
    general_chat,
    request_reference_clarification,
    request_guess_movie_unavailable,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from functools import partial
from app.agent.routing import (
    route_after_parse,
    route_after_resolve_references
)
from app.agent.state import AgentState
from app.config import settings

from app.services.movie_search import MovieSearchService
from app.services.embedder import embedder
from app.db.database import async_session_maker


def create_chat_model(temperature: float) -> ChatOpenAI:
    fallback_models = settings.llm_fallback_models
    openrouter_options = (
        {"extra_body": {"models": fallback_models}}
        if fallback_models
        else {}
    )

    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=temperature,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_URL,
        max_retries=3,
        **openrouter_options,
    )


llm_strict = create_chat_model(settings.LLM_STRICT_TEMPERATURE)
llm_creative = create_chat_model(settings.LLM_CREATIVE_TEMPERATURE)


movie_search_service = MovieSearchService(
    session_factory=async_session_maker,
    embedder=embedder,
)


workflow = StateGraph(AgentState)

workflow.add_node(
    "parser_query_node",
    partial(
        parser_query_node,
        llm=llm_strict
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
workflow.add_node(
    "evaluate_results",
    partial(
        evaluate_results,
        llm=llm_strict
    )
)
workflow.add_node(
    "compose_response",
    partial(
        compose_response,
        llm=llm_creative
    )
)
workflow.add_node(
    "request_reference_clarification",
    request_reference_clarification
)
workflow.add_node(
    "request_guess_movie_unavailable",
    request_guess_movie_unavailable,
)

workflow.set_entry_point("parser_query_node")
workflow.add_conditional_edges(
    "parser_query_node",
    route_after_parse,
    {
        "resolve_references": "resolve_references",
        "search_movies": "search_movies",
        "general_chat": "general_chat",
        "request_guess_movie_unavailable": "request_guess_movie_unavailable",
    }
)
workflow.add_conditional_edges(
    "resolve_references",
    route_after_resolve_references,
    {
        "request_reference_clarification": "request_reference_clarification",
        "search_movies": "search_movies",
    }
)
workflow.add_edge("search_movies", "evaluate_results")
workflow.add_edge("evaluate_results", "compose_response")
workflow.add_edge("request_reference_clarification", END)
workflow.add_edge("request_guess_movie_unavailable", END)
workflow.add_edge("compose_response", END)

graph = workflow.compile()
