import asyncio
from typing import List
from loguru import logger
from app.config import settings
from fastembed import TextEmbedding


class EmbedderService:
    def __init__(self, model_name: str):
        logger.info(f"EmbedderService запущен на ONNX Runtime (FastEmbed) | Модель: {model_name}")
        self.model = TextEmbedding(model_name=model_name)

    async def get_embedding(self, text: str) -> List[float]:
        def _encode() -> List[float]:
            gen = self.model.embed([text])
            vec = next(gen)
            return vec.tolist()

        return await asyncio.to_thread(_encode)

    async def get_embeddings(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        def _encode_batch() -> List[List[float]]:
            gen = self.model.embed(texts, batch_size=batch_size)
            return [vec.tolist() for vec in gen]

        return await asyncio.to_thread(_encode_batch)


embedder = EmbedderService(model_name=settings.EMBEDDER_MODEL_NAME)
