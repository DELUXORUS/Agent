import logging

from html import escape
from langchain_openai import ChatOpenAI

from app.schemas import MovieDTO
from langchain_core.messages import SystemMessage, HumanMessage

from app.agent.state import AgentState
from app.agent.schemas import (
    MovieQueryPlan,
    ResolvedMovieReference,
    ReferenceResolutionStatus,
    MovieEvaluation
)
from app.agent.load_prompts import prompts
from app.agent.mappers import build_movie_search_params
from app.services.schemas import MovieSearchParams
from app.services.movie_search import MovieSearchService


logger = logging.getLogger("uvicorn")


async def parser_query_node(
        state: AgentState,
        llm: ChatOpenAI
) -> dict:
    structed_output_llm = llm.with_structured_output(MovieQueryPlan)

    messages = [
        SystemMessage(content=prompts["SYSTEM_PARSER_QUERY_PROMPT"]),
        HumanMessage(content=state['user_query'])
    ]

    query_plan: MovieQueryPlan = await structed_output_llm.ainvoke(messages)

    return {
        'query_plan': query_plan,
    }


async def general_chat(
        state: AgentState,
        llm: ChatOpenAI
) -> dict:
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


async def resolve_references(
        state: AgentState,
        movie_search: MovieSearchService
) -> dict:
    query_plan = state["query_plan"]

    resolved_references = []
    for reference in query_plan.reference_movies:
        movies: list[MovieDTO] = await movie_search.resolve_reference(
            query=reference.query,
            exact_title=reference.exact_title,
            year=reference.year,
        )

        candidates = []
        if len(movies) > 1:
            status = ReferenceResolutionStatus.AMBIGUOUS
            candidates = movies
            movie = None
        elif len(movies) == 1:
            status = ReferenceResolutionStatus.RESOLVED
            movie = movies[0]
        else:
            movie = None
            status = ReferenceResolutionStatus.NOT_FOUND

        resolved_references.append(
            ResolvedMovieReference(
                reference=reference,
                movie=movie,
                status=status,
                candidates=candidates
            )
        )

    return {
        "resolved_references": resolved_references,
    }


async def request_reference_clarification(
    state: AgentState,
) -> dict:
    blocks: list[str] = []

    for resolved in state["resolved_references"]:
        query = escape(resolved.reference.query)

        if resolved.status == ReferenceResolutionStatus.AMBIGUOUS:
            lines = [
                f"Для «{query}» найдено несколько вариантов:"
            ]

            for movie in resolved.candidates:
                title = escape(movie.title)
                year = (
                    escape(movie.release_date[:4])
                    if movie.release_date
                    else "год неизвестен"
                )
                lines.append(f"• {title} — {year}")

            lines.append("Укажи год нужного фильма.")
            blocks.append("\n".join(lines))

        elif resolved.status == ReferenceResolutionStatus.NOT_FOUND:
            blocks.append(
                f"Не удалось найти в каталоге фильм «{query}».\n"
                "Проверь название или укажи оригинальное."
            )

    if not blocks:
        raise ValueError(
            "Clarification requires an unresolved reference"
        )

    blocks.append(
        "Повтори полный запрос с уточнениями для указанных фильмов."
    )

    return {
        "final_response": "\n\n".join(blocks),
    }


async def search_movies(
        state: AgentState,
        movie_search: MovieSearchService,
) -> dict:
    params: MovieSearchParams = build_movie_search_params(state)

    movies = await movie_search.search_recommendations(
        state['user_id'],
        params
    )

    return {
        "candidates": movies
    }


async def evaluate_results(
        state: AgentState,
        llm: ChatOpenAI
) -> dict:
    structed_output_llm = llm.with_structured_output(MovieEvaluation)

    messages = [
        SystemMessage(content=prompts["SYSTEM_PARSER_EVALUATE_PROMPT"]),
        HumanMessage(content=state['user_query'])
    ]

    response = structed_output_llm.ainvoke(messages)



async def compose_response(
        state: AgentState,
        llm: ChatOpenAI
) -> dict:
    pass
