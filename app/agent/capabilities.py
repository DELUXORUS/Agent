from app.agent.schemas import MovieQueryPlan


UNSUPPORTED_FILTER_LABELS = {
    "runtime": "хронометражу",
    "genres": "жанрам",
    "actors": "актёрам",
    "directors": "режиссёрам",
}


def get_unsupported_filters(query_plan: MovieQueryPlan) -> list[str]:
    unsupported_filters: list[str] = []

    if (
        query_plan.runtime_minutes.min is not None
        or query_plan.runtime_minutes.max is not None
    ):
        unsupported_filters.append("runtime")

    if query_plan.genres.include or query_plan.genres.exclude:
        unsupported_filters.append("genres")

    if query_plan.actors.include or query_plan.actors.exclude:
        unsupported_filters.append("actors")

    if query_plan.directors.include or query_plan.directors.exclude:
        unsupported_filters.append("directors")

    return unsupported_filters
