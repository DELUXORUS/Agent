from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from app.db.operations import Operations
from app.schemas import MovieDTO
from app.services.embedder import EmbedderService


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


    async def resolve_reference(
            self,
            query: str,
            exact_title: bool
    ) -> MovieDTO | None:
        pass


    async def guess(self):
        pass


    async def search_recommendations(
            self,
            user_id: int,
            *,
            query_vector: list[float] | None,
            limit: int,
            # фильтры добавим следующим шагом
    ) -> list[MovieDTO]:
        async with self._session_factory() as session:
            operations = Operations(session)

            watched_ids = await operations.get_user_watched_movie_ids(
                user_id
            )

            # Сюда позже перенесём вызов нового search(...)
            raise NotImplementedError
