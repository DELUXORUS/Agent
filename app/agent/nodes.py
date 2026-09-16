import logging
from langchain_openai import ChatOpenAI

from app.schemas import MovieDTO
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.schemas import (
    MovieQueryPlan,
    ResolvedMovieReference,
    ReferenceResolutionStatus
)
from app.agent.load_prompts import prompts
from app.services.movie_search import MovieSearchService
from app.services.schemas import MovieSearchParams


logger = logging.getLogger("uvicorn")


async def parser_query_node(state: AgentState, llm: ChatOpenAI) -> dict:
    structed_output_llm = llm.with_structured_output(MovieQueryPlan)

    messages = [
        SystemMessage(content=prompts["SYSTEM_PARSER_QUERY_PROMPT"]),
        HumanMessage(content=state['user_query'])
    ]

    query_plan: MovieQueryPlan = await structed_output_llm.ainvoke(messages)

    return {
        'query_plan': query_plan,
    }


async def general_chat(state: AgentState, llm: ChatOpenAI) -> dict:
    messages = [
        SystemMessage(
            content=(
                "Ты ассистент по фильмам и кино. "
                "Отвечай кратко, естественно и по существу."
            )
        ),
        HumanMessage(content=state["user_query"]),
    ]

    response = await llm.ainvoke(messages)

    return {
        "final_response": response.content,
    }


async def resolve_references(state: AgentState, movie_search: MovieSearchService) -> dict:
    query_plan = state["query_plan"]

    resolved_references = []
    for reference in query_plan.reference_movies:
        movie: MovieDTO | None = await movie_search.resolve_reference(
            query=reference.query,
            exact_title=reference.exact_title
        )

        status = (
            ReferenceResolutionStatus.RESOLVED
            if movie is not None
            else ReferenceResolutionStatus.NOT_FOUND
        )

        resolved_references.append(
            ResolvedMovieReference(
                reference=reference,
                movie=movie,
                status=status,
            )
        )

    return {
        "resolved_references": resolved_references,
    }


async def search_movies(state: AgentState) -> dict:
    pass

async def evaluate_results(state: AgentState, llm: ChatOpenAI) -> dict:
    pass

async def compose_response(state: AgentState, llm: ChatOpenAI) -> dict:
    pass