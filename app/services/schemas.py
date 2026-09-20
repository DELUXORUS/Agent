from dataclasses import dataclass, field


@dataclass(slots=True)
class MovieSearchParams:
    genres: list[str] = field(default_factory=list)
    excluded_genres: list[str] = field(default_factory=list)

    actors: list[str] = field(default_factory=list)
    directors: list[str] = field(default_factory=list)

    year_min: int | None = None
    year_max: int | None = None

    rating_min: float | None = None
    rating_max: float | None = None

    runtime_min: int | None = None
    runtime_max: int | None = None

    semantic_query: str | None = None

    similar_movie_ids: list[int] = field(default_factory=list)
    excluded_movie_ids: list[int] = field(default_factory=list)

    excluded_actors: list[str] = field(default_factory=list)
    excluded_directors: list[str] = field(default_factory=list)

    limit: int = 5