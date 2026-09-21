from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from app.schemas import MovieDTO


class Intent(str, Enum):
    GUESS_MOVIE = "guess_movie"
    RECOMMEND_MOVIES = "recommend_movies"
    GENERAL_CHAT = "general_chat"


class MovieEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted_movie_ids: list[StrictInt] = Field(
        description="IDs of relevant candidates; empty when none are suitable."
    )

    @field_validator("accepted_movie_ids")
    @classmethod
    def duplicate_ids(cls, ids: list[int]) -> list[int]:
        if len(ids) != len(set(ids)):
            dict_ids = dict.fromkeys(ids)
            ids = list(dict_ids.keys())
        return ids


class ReferenceRelation(str, Enum):
    SIMILAR_TO = "similar_to"
    EXCLUDE = "exclude"


class ReferenceResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    REQUIRES_EXACT_TITLE = "requires_exact_title"


class EntityFilter(BaseModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class IntRange(BaseModel):
    min: int | None = None
    max: int | None = None


class FloatRange(BaseModel):
    min: float | None = None
    max: float | None = None


class MovieReference(BaseModel):
    query: str
    exact_title: bool = False
    year: int | None = None
    relation: ReferenceRelation = ReferenceRelation.SIMILAR_TO

class ResolvedMovieReference(BaseModel):
    reference: MovieReference
    movie: MovieDTO | None = None
    status: ReferenceResolutionStatus
    candidates: list[MovieDTO] = Field(default_factory=list)


class MovieQueryPlan(BaseModel):
    intent: Intent

    reference_movies: list[MovieReference] = Field(
        default_factory=list
    )

    genres: EntityFilter = Field(default_factory=EntityFilter)
    actors: EntityFilter = Field(default_factory=EntityFilter)
    directors: EntityFilter = Field(default_factory=EntityFilter)

    year: IntRange = Field(default_factory=IntRange)
    rating: FloatRange = Field(default_factory=FloatRange)
    runtime_minutes: IntRange = Field(default_factory=IntRange)

    desired_traits: list[str] = Field(default_factory=list)
    undesired_traits: list[str] = Field(default_factory=list)

    semantic_query: str | None = None

    result_limit: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of movies shown to the user.",
    )
