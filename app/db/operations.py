from sqlalchemy import select
from datetime import date

from app.schemas import MovieDTO

from app.db.models import Movie
from app.db.models import UserMovieHistory
from app.db.filters import MovieFilters, apply_movie_filters

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.dialects.postgresql import insert


def build_movie_conditions(
    filters: MovieFilters,
) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = []

    if filters.year_min is not None:
        conditions.append(
            Movie.release_date >= date(filters.year_min, 1, 1)
        )

    if filters.year_max is not None:
        conditions.append(
            Movie.release_date <= date(filters.year_max, 12, 31)
        )

    if filters.rating_min is not None:
        conditions.append(
            Movie.vote_average >= filters.rating_min
        )

    if filters.rating_max is not None:
        conditions.append(
            Movie.vote_average <= filters.rating_max
        )

    if filters.excluded_movie_ids:
        conditions.append(
            Movie.id.not_in(filters.excluded_movie_ids)
        )

    return conditions



class Operations:
    def __init__(self, session: AsyncSession):
        self.session = session


    async def find_movies_by_title(
        self,
        title: str,
        year: int | None = None,
    ) -> list[MovieDTO]:
        stmt = (
            select(Movie)
            .where(Movie.title == title)
        )

        if year is not None:
            stmt = stmt.where(
                Movie.release_date >= date(year, 1, 1),
                Movie.release_date <= date(year, 12, 31),
            )

        stmt = stmt.order_by(Movie.release_date.asc().nulls_last(), Movie.id.asc())

        result = await self.session.execute(stmt)

        movies = result.scalars().all()

        return [MovieDTO.model_validate(movie) for movie in movies]


    async def get_watched_movie_ids(
            self,
            user_id: int
    ) -> list[int]:
        stmt = (
            select(UserMovieHistory.movie_id)
            .where(UserMovieHistory.user_id == user_id)
        )

        result = await self.session.execute(stmt)

        return list(result.scalars().all())


# TODO: Реализовать поиск по несколькоим столбцам с эмбеддингами. На данный момент он один
    async def get_movie_embeddings(
        self,
        movie_ids: list[int],
    ) -> dict[int, list[float]]:
        stmt = (
            select(Movie.id, Movie.embedding)
            .where(
                Movie.id.in_(movie_ids),
                Movie.embedding.is_not(None),
            )
        )

        result = await self.session.execute(stmt)

        return {
            movie_id: list(embedding)
            for movie_id, embedding in result.all()
        }


    async def search_movies(
        self,
        filters: MovieFilters,
        query_embedding: list[float] | None,
        limit: int,
    ) -> list[MovieDTO]:
        if limit <= 0:
            raise ValueError("limit must be positive")

        stmt = apply_movie_filters(select(Movie), filters)

        if query_embedding is not None:
            stmt = (
                stmt
                .where(Movie.embedding.is_not(None))
                .order_by(
                    Movie.embedding.cosine_distance(query_embedding),
                    Movie.id.asc(),
                )
            )
        else:
            stmt = stmt.order_by(
                Movie.vote_average.desc().nulls_last(),
                Movie.id.asc(),
            )

        stmt = stmt.limit(limit)
        result = await self.session.execute(stmt)
        movies = result.scalars().all()

        return [
            MovieDTO.model_validate(movie)
            for movie in movies
        ]


    async def add_movies_to_user_history(
            self,
            user_id: int,
            movie_id: int
    ):
        stmt = insert(UserMovieHistory).values(
                user_id=user_id,
                movie_id=movie_id,
            )

        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_user_movies_history_user_movie"
        )

        await self.session.execute(stmt)
        await self.session.commit()
