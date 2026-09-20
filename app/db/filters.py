from dataclasses import dataclass, field
from datetime import date

from sqlalchemy import Select

from app.db.models import Movie


@dataclass(slots=True)
class MovieFilters:
    year_min: int | None = None
    year_max: int | None = None

    rating_min: float | None = None
    rating_max: float | None = None

    runtime_min: int | None = None
    runtime_max: int | None = None

    included_genres: list[str] = field(default_factory=list)
    excluded_genres: list[str] = field(default_factory=list)

    included_actors: list[str] = field(default_factory=list)
    excluded_actors: list[str] = field(default_factory=list)

    included_directors: list[str] = field(default_factory=list)
    excluded_directors: list[str] = field(default_factory=list)

    excluded_movie_ids: list[int] = field(default_factory=list)


def apply_movie_filters(
    stmt: Select[tuple[Movie]],
    filters: MovieFilters,
) -> Select[tuple[Movie]]:
    if filters.runtime_min is not None:
        stmt = stmt.where(Movie.runtime >= filters.runtime_min)
    if filters.runtime_max is not None:
        stmt = stmt.where(Movie.runtime <= filters.runtime_max)

    for column, included, excluded in (
        (Movie.genres, filters.included_genres, filters.excluded_genres),
        (Movie.actors, filters.included_actors, filters.excluded_actors),
        (Movie.directors, filters.included_directors, filters.excluded_directors),
    ):
        if included:
            stmt = stmt.where(column.contains(included))
        if excluded:
            stmt = stmt.where(~column.overlap(excluded))

    if filters.year_min is not None:
        stmt = stmt.where(
            Movie.release_date >= date(filters.year_min, 1, 1)
        )

    if filters.year_max is not None:
        stmt = stmt.where(
            Movie.release_date <= date(filters.year_max, 12, 31)
        )

    if filters.rating_min is not None:
        stmt = stmt.where(Movie.vote_average >= filters.rating_min)

    if filters.rating_max is not None:
        stmt = stmt.where(Movie.vote_average <= filters.rating_max)

    if filters.excluded_movie_ids:
        stmt = stmt.where(Movie.id.not_in(filters.excluded_movie_ids))

    return stmt
