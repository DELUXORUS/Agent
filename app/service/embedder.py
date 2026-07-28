import torch
from sentence_transformers import SentenceTransformer


class EmbedderService:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"EmbedderService запущен на устройстве: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(self, text: str) -> list[float]:
        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding.tolist()

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=256,  # Оптимальный размер батча для GPU
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


embedder = EmbedderService()

# from fastembed import TextEmbedding
#
#
# class EmbedderService:
#     def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
#         print(f"EmbedderService запущен на ONNX Runtime (FastEmbed)...")
#         # FastEmbed автоматически скачивает и запускает квантованную/ONNX версию модели
#         self.model = TextEmbedding(model_name=model_name)
#
#     def get_embedding(self, text: str) -> list[float]:
#         # fastembed принимает список и возвращает генератор, берем первый вектор
#         embeddings_generator = self.model.embed([text])
#         return next(embeddings_generator).tolist()
#
#     def get_embeddings(self, texts: list[str]) -> list[list[float]]:
#         # embed преобразует батч текстов в генератор numpy-массивов
#         embeddings_generator = self.model.embed(texts, batch_size=256)
#         return [vec.tolist() for vec in embeddings_generator]
#
#
# embedder = EmbedderService()