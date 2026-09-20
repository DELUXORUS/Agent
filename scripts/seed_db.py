import asyncio
from typing import TYPE_CHECKING

from scripts.movie_dataset import (
    MovieSeedRecord,
    build_movie_embedding_text,
    build_movie_seed_records,
    merge_movie_datasets,
    prepare_and_filter_movies,
)

if TYPE_CHECKING:
    from app.services.embedder import EmbedderService
from app.constants import EMBEDDING_DIMENSION


async def generate_movie_embeddings(
    movies: list[MovieSeedRecord],
    embedder: "EmbedderService",
) -> list[list[float]]:
    if not movies:
        return []

    semantic_movie_texts = [
        build_movie_embedding_text(movie)
        for movie in movies
    ]

    embeddings = await embedder.get_embeddings(
        semantic_movie_texts,
        batch_size=32,
    )

    if len(embeddings) != len(movies):
        raise ValueError(
            f"Expected {len(movies)} embeddings, "
            f"got {len(embeddings)}"
        )

    for index, embedding in enumerate(embeddings):
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding at index {index}: "
                f"expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {len(embedding)}"
            )

    return embeddings


async def main() -> None:
    """Smoke-test three real movies; no database writes."""
    from scripts.inspect_movie_dataset import load_raw_datasets
    from app.services.embedder import embedder

    movies, credits, keywords = load_raw_datasets()
    merged = merge_movie_datasets(movies, credits, keywords)
    prepared, _ = prepare_and_filter_movies(merged)
    sample_movies = build_movie_seed_records(prepared.head(3))
    if len(sample_movies) != 3:
        raise ValueError("Smoke test requires at least three prepared movies")

    embeddings = await generate_movie_embeddings(sample_movies, embedder)
    for movie, embedding in zip(sample_movies, embeddings, strict=True):
        print(f"{movie.title}: dimensions={len(embedding)}")


if __name__ == "__main__":
    asyncio.run(main())
