from app.db.models import Movie
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import UserMovieHistory
from app.schemas import MovieDTO


class Operations:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_movies_count(self) -> int:
        stmt = select(func.count()).select_from(Movie)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_user_watched_movies(self, user_id: int) -> list(MovieDTO):
        stmt = (
            select(Movie)
            .join(UserMovieHistory, UserMovieHistory.movie_id == Movie.id)
            .where(UserMovieHistory.user_id == user_id)
        )

        result = await self.session.execute(stmt)
        movies_orm = result.scalars().all()

        return [MovieDTO.model_validate(movie) for movie in movies_orm]

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
    ) -> list(MovieDTO):
        stmt = select(Movie)

        if exclude_movie_ids:
            stmt = stmt.where(Movie.id.not_in(exclude_movie_ids))

        if min_vote_average is not None:
            stmt = stmt.where(Movie.vote_average >= min_vote_average)

        if genre:
            stmt = stmt.where(Movie.genres.ilike(f"%{genre}%"))

        if query_vector is not None:
            stmt = stmt.order_by(Movie.embedding.cosine_distance(query_vector))
        else:
            stmt = stmt.order_by(Movie.vote_average.desc())

        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        movies_orm = result.scalars().all()

        return [MovieDTO.model_validate(movie) for movie in movies_orm]

    async def search_hybrid_guess(
            self,
            query_vector: list[float],
            text_query: str,
            limit: int = 3,
            k: int = 60,  # Стандартная константа для RRF
    ) -> list[MovieDTO]:
        # 1. Достаем Top-10 кандидатов по векторному поиску
        vector_stmt = (
            select(Movie)
            .order_by(Movie.embedding.cosine_distance(query_vector))
            .limit(10)
        )
        vector_res = await self.session.execute(vector_stmt)
        vector_movies = list(vector_res.scalars().all())

        # 2. Достаем Top-10 кандидатов по ключевым словам / полнотекстовому поиску
        words = [w.strip() for w in text_query.split() if len(w.strip()) > 3]
        text_movies = []
        if words:
            ilike_conditions = [
                or_(
                    Movie.title.ilike(f"%{word}%"),
                    Movie.overview.ilike(f"%{word}%")
                )
                for word in words[:3]
            ]
            text_stmt = (
                select(Movie)
                .where(or_(*ilike_conditions))
                .order_by(Movie.vote_average.desc())
                .limit(10)
            )
            text_res = await self.session.execute(text_stmt)
            text_movies = list(text_res.scalars().all())

        # 3. Применяем алгоритм Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[int, float] = {}
        movies_map: dict[int, Movie] = {}

        # Считаем ранги для векторного поиска
        for rank, movie in enumerate(vector_movies, start=1):
            movies_map[movie.id] = movie
            rrf_scores[movie.id] = rrf_scores.get(movie.id, 0.0) + (1.0 / (k + rank))

        # Считаем ранги для полнотекстового поиска
        for rank, movie in enumerate(text_movies, start=1):
            movies_map[movie.id] = movie
            rrf_scores[movie.id] = rrf_scores.get(movie.id, 0.0) + (1.0 / (k + rank))

        # 4. Сортируем все уникальные фильмы по убыванию RRF score
        sorted_movie_ids = sorted(
            rrf_scores.keys(),
            key=lambda movie_id: rrf_scores[movie_id],
            reverse=True
        )

        # Берем top-k результатов и конвертируем в DTO
        top_movies = [movies_map[m_id] for m_id in sorted_movie_ids[:limit]]
        return [MovieDTO.model_validate(movie) for movie in top_movies]

