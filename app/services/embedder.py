import asyncio

from loguru import logger
from fastembed import TextEmbedding

from app.config import settings
from app.constants import EMBEDDING_DIMENSION


class EmbeddingDimensionError(ValueError):
    pass


class EmbedderService:
    def __init__(self, model_name: str):
        model_dimension = TextEmbedding.get_embedding_size(model_name)
        if model_dimension != EMBEDDING_DIMENSION:
            raise EmbeddingDimensionError(
                f"Embedding model '{model_name}' produces "
                f"{model_dimension}-dimensional vectors, but the database "
                f"expects {EMBEDDING_DIMENSION}"
            )

        logger.info(
            "EmbedderService запущен на ONNX Runtime (FastEmbed) | "
            f"Модель: {model_name} | Размерность: {model_dimension}"
        )
        self.model = TextEmbedding(model_name=model_name)

    async def get_embedding(self, text: str) -> list[float]:
        def _encode() -> list[float]:
            gen = self.model.embed([text])
            vec = next(gen)
            return vec.tolist()

        return await asyncio.to_thread(_encode)

    async def get_embeddings(
        self,
        texts: list[str],
        batch_size: int = 256,
    ) -> list[list[float]]:
        def _encode_batch() -> list[list[float]]:
            gen = self.model.embed(texts, batch_size=batch_size)
            return [vec.tolist() for vec in gen]

        return await asyncio.to_thread(_encode_batch)


embedder = EmbedderService(model_name=settings.EMBEDDER_MODEL_NAME)
