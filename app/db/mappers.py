from app.core.text_normalization import normalize_search_text
from app.db.filters import MovieFilters
from app.services.schemas import MovieSearchParams


def normalize_filter_values(values: list[str]) -> list[str]:
    normalized = (normalize_search_text(value) for value in values)
    return list(dict.fromkeys(value for value in normalized if value))


def build_movie_filters(params: MovieSearchParams) -> MovieFilters:
    return MovieFilters(
        year_min=params.year_min,
        year_max=params.year_max,
        rating_min=params.rating_min,
        rating_max=params.rating_max,
        runtime_min=params.runtime_min,
        runtime_max=params.runtime_max,
        included_genres=normalize_filter_values(params.genres),
        excluded_genres=normalize_filter_values(params.excluded_genres),
        included_actors=normalize_filter_values(params.actors),
        included_directors=normalize_filter_values(params.directors),
        excluded_actors=normalize_filter_values(params.excluded_actors),
        excluded_directors=normalize_filter_values(params.excluded_directors),
        excluded_movie_ids=list(params.excluded_movie_ids),
    )