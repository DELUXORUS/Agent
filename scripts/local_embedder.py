import torch
import asyncio
from sentence_transformers import SentenceTransformer

class LocalGPUEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Локальный сид запущен на устройстве: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    async def get_embeddings(self, texts: list[str], batch_size: int = 512) -> list[list[float]]:
        def _encode():
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            ).tolist()

        return await asyncio.to_thread(_encode)

embedder = LocalGPUEmbedder()