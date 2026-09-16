from app.db.filters import MovieFilters
from app.services.schemas import MovieSearchParams


def build_movie_filters(params: MovieSearchParams) -> MovieFilters:
    return MovieFilters(
        year_min=params.year_min,
        year_max=params.year_max,
        rating_min=params.rating_min,
        rating_max=params.rating_max,
        runtime_min=params.runtime_min,
        runtime_max=params.runtime_max,
        included_genres=list(params.genres),
        excluded_genres=list(params.excluded_genres),
        included_actors=list(params.actors),
        included_directors=list(params.directors),
        excluded_movie_ids=list(params.excluded_movie_ids),
    )