from enum import Enum
from pydantic import BaseModel, Field

from app.schemas import MovieDTO


class Intent(str, Enum):
    GUESS_MOVIE = "guess_movie"
    RECOMMEND_MOVIES = "recommend_movies"
    GENERAL_CHAT = "general_chat"


class ReferenceRelation(str, Enum):
    SIMILAR_TO = "similar_to"
    EXCLUDE = "exclude"


class ReferenceResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    NOT_FOUND = "not_found"


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
    relation: ReferenceRelation = ReferenceRelation.SIMILAR_TO

class ResolvedMovieReference(BaseModel):
    reference: MovieReference
    movie: MovieDTO | None = None
    status: ReferenceResolutionStatus


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

    limit: int = Field(default=5, ge=1, le=10)