import logging
import json

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


async def request_guess_movie_unavailable(
    state: AgentState,
) -> dict:
    return {
        "final_response": (
            "Поиск фильма по описанию пока не поддерживается. "
            "Укажи точное название фильма, если оно тебе известно."
        ),
    }


async def resolve_references(
        state: AgentState,
        movie_search: MovieSearchService
) -> dict:
    query_plan = state["query_plan"]

    resolved_references = []
    for reference in query_plan.reference_movies:
        if not reference.exact_title:
            resolved_references.append(
                ResolvedMovieReference(
                    reference=reference,
                    movie=None,
                    status=(
                        ReferenceResolutionStatus.REQUIRES_EXACT_TITLE
                    ),
                )
            )
            continue

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
                year = str(movie.release_date.year)
                lines.append(f"• {title} — {year}")

            lines.append("Укажи год нужного фильма.")
            blocks.append("\n".join(lines))

        elif resolved.status == ReferenceResolutionStatus.NOT_FOUND:
            blocks.append(
                f"Не удалось найти в каталоге фильм «{query}».\n"
                "Проверь название или укажи оригинальное."
            )

        elif (
            resolved.status
            == ReferenceResolutionStatus.REQUIRES_EXACT_TITLE
        ):
            blocks.append(
                f"Не удалось однозначно определить фильм по описанию «{query}».\n"
                "Укажи точное название фильма."
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
    candidates = state["candidates"]

    if not candidates:
        return {"selected_movies": []}

    payload = {
        "user_query": state["user_query"],
        "candidates": [
            {
                "id": movie.id,
                "title": movie.title,
                "original_title": movie.original_title,
                "overview": movie.overview,
                "tagline": movie.tagline,
                "genres": movie.genres,
                "release_date": movie.release_date.isoformat(),
                "runtime": movie.runtime,
                "vote_average": movie.vote_average,
                "keywords": movie.keywords[:25],
            }
            for movie in candidates
        ],
    }

    messages = [
        SystemMessage(
            content=prompts["SYSTEM_PARSER_EVALUATE_PROMPT"]
        ),
        HumanMessage(
            content=json.dumps(payload, ensure_ascii=False)
        ),
    ]

    structed_output_llm = llm.with_structured_output(MovieEvaluation)
    response = await structed_output_llm.ainvoke(messages)

    evaluation = MovieEvaluation.model_validate(response)

    accepted_ids = set(evaluation.accepted_movie_ids)
    candidate_ids = {movie.id for movie in candidates}

    if not accepted_ids.issubset(candidate_ids):
        raise ValueError(
            "Evaluation returned IDs outside the candidate list"
        )

    selected_movies = [
        movie
        for movie in candidates
        if movie.id in accepted_ids
    ]

    return {"selected_movies": selected_movies}



async def compose_response(
        state: AgentState,
        llm: ChatOpenAI
) -> dict:
    selected_movies = state["selected_movies"]

    if selected_movies:
        response = "Вот подходящие фильмы:\n\n"

        for index, movie in enumerate(selected_movies, start=1):
            title = escape(movie.title)
            year = str(movie.release_date.year)
            rating = movie.vote_average

            response += f"{index}. {title} — {year} · рейтинг: {rating}\n"
    else:
        response = "В нашем каталоге не удалось найти подходящие фильмы."

    return {"final_response": response.strip()}
