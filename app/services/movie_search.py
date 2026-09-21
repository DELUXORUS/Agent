from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.db.mappers import build_movie_filters
from app.db.operations import Operations
from app.schemas import MovieDTO
if TYPE_CHECKING:
    from app.services.embedder import EmbedderService
from app.services.schemas import MovieSearchParams


class MissingMovieEmbeddingsError(RuntimeError):
    def __init__(self, movie_ids: set[int]):
        self.movie_ids = tuple(sorted(movie_ids))
        super().__init__(
            f"Movies have no embeddings: {', '.join(map(str, self.movie_ids))}"
        )


def mean_embeddings(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("At least one embedding is required")

    dimensions = {len(embedding) for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("Embeddings must have the same dimensions")

    return np.mean(embeddings, axis=0).tolist()


class MovieSearchService:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: EmbedderService,
    ):
        self._session_factory = session_factory
        self._embedder = embedder


    async def get_embedding(self, query: str) -> list[float]:
        return await self._embedder.get_embedding(query)

# TODO Сделать разрешение ссылок на фильмы не только по названию, но и по описанию
    async def resolve_reference(
            self,
            query: str,
            exact_title: bool,
            year: int | None = None,
    ) -> list[MovieDTO]:
        async with self._session_factory() as session:
            operation = Operations(session)

            if not exact_title:
                raise NotImplementedError(
                    "Reference resolution by description is not implemented yet"
                )

            result = await operation.find_movies_by_title(query, year=year)

            return result

    async def search_recommendations(
            self,
            user_id: int,
            params: MovieSearchParams
    ) -> list[MovieDTO]:
        filters = build_movie_filters(params)

        semantic_embedding = None
        if params.semantic_query:
            semantic_embedding = await self._embedder.get_embedding(
                params.semantic_query
            )

        async with self._session_factory() as session:
            operations = Operations(session)

            reference_embedding = None
            similar_movie_ids = set(params.similar_movie_ids)

            if similar_movie_ids:
                embeddings_by_id = await operations.get_movie_embeddings(
                    list(similar_movie_ids)
                )
                missing_movie_ids = similar_movie_ids - embeddings_by_id.keys()
                if missing_movie_ids:
                    raise MissingMovieEmbeddingsError(missing_movie_ids)

                reference_embedding = mean_embeddings(
                    list(embeddings_by_id.values())
                )

            if semantic_embedding is not None and reference_embedding is not None:
                result_embedding = mean_embeddings([
                    semantic_embedding,
                    reference_embedding,
                ])
            elif reference_embedding is not None:
                result_embedding = reference_embedding
            else:
                result_embedding = semantic_embedding

            watched_ids = await operations.get_watched_movie_ids(
                user_id
            )

            filters.excluded_movie_ids = list(
                set(filters.excluded_movie_ids)
                | set(watched_ids)
                | similar_movie_ids
            )

            return await operations.search_movies(
                filters=filters,
                query_embedding=result_embedding,
                limit=params.candidate_limit,
            )



