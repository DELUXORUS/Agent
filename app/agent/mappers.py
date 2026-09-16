from app.agent.schemas import ReferenceRelation
from app.agent.state import AgentState
from app.services.schemas import MovieSearchParams


def build_movie_search_params(
    state: AgentState,
) -> MovieSearchParams:
    query_plan = state["query_plan"]

    similar_movie_ids: list[int] = []
    excluded_movie_ids: list[int] = []

    for resolved in state.get("resolved_references", []):
        if resolved.movie is None:
            continue

        if resolved.reference.relation == ReferenceRelation.SIMILAR_TO:
            similar_movie_ids.append(resolved.movie.id)

        elif resolved.reference.relation == ReferenceRelation.EXCLUDE:
            excluded_movie_ids.append(resolved.movie.id)

    return MovieSearchParams(
        genres=query_plan.genres.include,
        excluded_genres=query_plan.genres.exclude,
        actors=query_plan.actors.include,
        directors=query_plan.directors.include,

        year_min=query_plan.year.min,
        year_max=query_plan.year.max,

        rating_min=query_plan.rating.min,
        rating_max=query_plan.rating.max,

        runtime_min=query_plan.runtime_minutes.min,
        runtime_max=query_plan.runtime_minutes.max,

        semantic_query=query_plan.semantic_query,

        similar_movie_ids=similar_movie_ids,
        excluded_movie_ids=excluded_movie_ids,

        limit=query_plan.limit,
    )

