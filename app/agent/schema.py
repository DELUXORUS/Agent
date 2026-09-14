from pydantic import (
    BaseModel, Field,
    Optional, Enum
)


class Intent(str, Enum):
    GUESS_MOVIE = "guess_movie"
    RECOMMEND_MOVIES = "recommend_movies"
    GENERAL_CHAT = "general_chat"


class EntityFilter(BaseModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class NumericRange(BaseModel):
    min: float | None = None
    max: float | None = None


class MovieReference(BaseModel):
    query: str
    exact_title: bool = False


class MovieQueryPlan(BaseModel):
    intent: Intent

    references_movie: list[MovieReference] = Field(default_factory=list)

    genres: EntityFilter = Field(default_factory=EntityFilter)
    actors: EntityFilter = Field(default_factory=EntityFilter)
    directors: EntityFilter = Field(default_factory=EntityFilter)

    year: NumericRange | None = None
    rating: NumericRange | None = None
    runtime_minutes: NumericRange | None = None

    desired_traits: list[str] = Field(default_factory=list)
    undesired_traits: list[str] = Field(default_factory=list)

    semantic_query: str | None = None

    limit: int = Field(default=5, ge=1, le=10)