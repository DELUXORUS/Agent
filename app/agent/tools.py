from app.service.embedder import embedder
from app.db.operations import Operations
from app.db.database import async_session_maker
from app.db.models import Movie


def generate_query_embedding(query: str) -> list[float]:
    return embedder.get_embedding(query)

async def fetch_recommended_movies(
    user_id: int,
    query_vector: list[float] | None = None,
    limit: int = 3,
    min_vote_average: float | None = None,
    genre: str | None = None,
) -> list[Movie]:
    async with async_session_maker() as session:  #
        operation = Operations(session)  #

        watched_movies = await operation.get_user_watched_movies(user_id)
        watched_ids = [movie.id for movie in watched_movies] if watched_movies else []

        return await operation.search_similar_movies_with_filters(  #
            query_vector=query_vector,
            limit=limit,
            min_vote_average=min_vote_average,
            genre=genre,
            exclude_movie_ids=watched_ids,
        )

async def fetch_guessed_movie_hybrid(
    query_vector: list[float],
    raw_query_text: str,
) -> list[Movie]:
    async with async_session_maker() as session:
        ops = Operations(session)
        return await ops.search_hybrid_guess(
            query_vector=query_vector,
            text_query=raw_query_text,
            limit=1,
        )