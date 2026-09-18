from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.db.mappers import build_movie_filters
from app.db.operations import Operations
from app.schemas import MovieDTO
if TYPE_CHECKING:
    from app.services.embedder import EmbedderService
from app.services.schemas import MovieSearchParams


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


    async def search_for_guess(
            self,
            user_id: int,
            movie_searhc_params: MovieSearchParams
    ) -> MovieDTO | None:
        pass


    async def search_recommendations(
            self,
            user_id: int,
            params: MovieSearchParams
    ) -> list[MovieDTO]:
        filters = build_movie_filters(params)

        query_embedding = None

        if params.semantic_query:
            query_embedding = await self._embedder.get_embedding(
                params.semantic_query
            )

        async with self._session_factory() as session:
            operations = Operations(session)

            watched_ids = await operations.get_watched_movie_ids(
                user_id
            )

            filters.excluded_movie_ids = list(
                set(filters.excluded_movie_ids) | set(watched_ids)
            )

            return await operations.search_movies(
                filters=filters,
                query_embedding=query_embedding,
                limit=params.limit,
            )



