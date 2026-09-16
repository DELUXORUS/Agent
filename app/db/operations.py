from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import MovieDTO
from app.db.models import Movie
from app.db.models import UserMovieHistory


#ОТРЕДАКТИРОВАТЬ


class Operations:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_movies_count(self) -> int:
        stmt = select(func.count()).select_from(Movie)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_user_watched_movie_ids(self, user_id: int) -> list[int]:
        stmt = select(UserMovieHistory.movie_id).where(UserMovieHistory.user_id == user_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_user_watched_movies(self, user_id: int) -> list[MovieDTO]:
        stmt = (
            select(Movie)
            .join(UserMovieHistory, UserMovieHistory.movie_id == Movie.id)
            .where(UserMovieHistory.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        movies_orm = result.scalars().all()
        return [MovieDTO.model_validate(movie) for movie in movies_orm]

    async def add_movies_to_user_history(
            self, user_id: int, movie_ids: list[int]
    ) -> None:
        if not movie_ids:
            return

        existing_ids = set(await self.get_user_watched_movie_ids(user_id))

        new_records = [
            UserMovieHistory(user_id=user_id, movie_id=m_id)
            for m_id in movie_ids
            if m_id not in existing_ids
        ]

        if new_records:
            self.session.add_all(new_records)
            await self.session.commit()

    async def insert_movies_batch(self, movies: list[Movie]) -> None:
        self.session.add_all(movies)
        await self.session.commit()
        self.session.expunge_all()

    async def search_similar_movies_with_filters(self) -> list[MovieDTO]:                                                #ЗАМЕНИТЬ ILIKE
        pass

    async def search_hybrid_guess(self) -> list[MovieDTO]:
        pass
