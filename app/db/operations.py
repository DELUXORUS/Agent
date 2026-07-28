from app.db.models import Movie
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import UserMovieHistory


class Operations:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_movies_count(self) -> int:
        stmt = select(func.count()).select_from(Movie)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_user_watched_movies(self, user_id: int) -> list[Movie]:
        stmt = (
            select(Movie)
            .join(UserMovieHistory, UserMovieHistory.movie_id == Movie.id)
            .where(UserMovieHistory.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def insert_movies_batch(self, movies: list[Movie]) -> None:
        self.session.add_all(movies)
        await self.session.commit()
        self.session.expunge_all()

    async def search_similar_movies_with_filters(
            self,
            query_vector: list[float] | None = None,
            limit: int = 3,
            min_vote_average: float | None = None,
            genre: str | None = None,
            exclude_movie_ids: list[int] | None = None,
    ) -> list[Movie]:
        stmt = select(Movie)

        # 1. Исключаем просмотренные фильмы по их ID
        if exclude_movie_ids:
            stmt = stmt.where(Movie.id.not_in(exclude_movie_ids))

        # 2. Фильтр по минимальному рейтингу
        if min_vote_average is not None:
            stmt = stmt.where(Movie.vote_average >= min_vote_average)

        # 3. Фильтр по жанру
        if genre:
            stmt = stmt.where(Movie.genres.ilike(f"%{genre}%"))

        # 4. Сортировка: по косинусному расстоянию или по рейтингу
        if query_vector is not None:
            stmt = stmt.order_by(Movie.embedding.cosine_distance(query_vector))
        else:
            stmt = stmt.order_by(Movie.vote_average.desc())

        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def search_hybrid_guess(
            self,
            query_vector: list[float],
            text_query: str,
            limit: int = 1,
    ) -> list[Movie]:
        # 1. Поиск по вектору (Semantic Search)
        vector_stmt = (
            select(Movie)
            .order_by(Movie.embedding.cosine_distance(query_vector))
            .limit(limit)
        )
        vector_res = await self.session.execute(vector_stmt)
        vector_movies = list(vector_res.scalars().all())

        # 2. Полнотекстовый / Ключевой поиск (Full-Text Search)
        # Формируем поисковые слова для ILIKE / TSQUERY из текста пользователя
        words = [w.strip() for w in text_query.split() if len(w.strip()) > 3]

        text_movies = []
        if words:
            # Ищем совпадения ключевых слов в названии или описании
            ilike_conditions = [
                or_(
                    Movie.title.ilike(f"%{word}%"),
                    Movie.overview.ilike(f"%{word}%")
                )
                for word in words[:3]  # Берем первые 3 ключевых слова
            ]

            text_stmt = (
                select(Movie)
                .where(or_(*ilike_conditions))
                .order_by(Movie.vote_average.desc())
                .limit(limit)
            )
            text_res = await self.session.execute(text_stmt)
            text_movies = list(text_res.scalars().all())

        # 3. Дедупликация и ранжирование (Векторный поиск имеет приоритет)
        seen_ids = set()
        final_movies = []

        for movie in vector_movies + text_movies:
            if movie.id not in seen_ids:
                seen_ids.add(movie.id)
                final_movies.append(movie)

        # Возвращаем top-k результатов (по умолчанию limit=1)
        return final_movies[:limit]
