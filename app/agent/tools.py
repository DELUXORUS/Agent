from app.agent.state import MovieFilter
from app.db.database import async_session_maker
from app.db.operations import Operations
from app.schemas import MovieDTO
from app.services.embedder import embedder


async def generate_query_embedding(query: str) -> list[float]:
    return await embedder.get_embedding(query)


async def fetch_recommended_movies(
    user_id: int,
    movie_filter: MovieFilter,
    query_vector: list[float] | None = None,
    limit: int = 3,
) -> list[MovieDTO]:
    async with async_session_maker() as session:
        operation = Operations(session)

        watched_ids = await operation.get_user_watched_movie_ids(user_id)

        return await operation.search_similar_movies_with_filters(
            movie_filter=movie_filter,
            query_vector=query_vector,
            limit=limit,
            exclude_movie_ids=watched_ids if watched_ids else None,
        )


async def fetch_guessed_movie_hybrid(
    query_vector: list[float],
    raw_query_text: str,
) -> list[MovieDTO]:
    async with async_session_maker() as session:
        ops = Operations(session)
        return await ops.search_hybrid_guess(
            query_vector=query_vector,
            text_query=raw_query_text,
            limit=1,
        )