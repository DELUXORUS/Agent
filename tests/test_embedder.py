import runpy
from pathlib import Path

import pytest


def load_embedder_module(monkeypatch):
    created_models: list[str] = []

    class FakeTextEmbedding:
        @staticmethod
        def get_embedding_size(model_name: str) -> int:
            if model_name == "wrong-dimension-model":
                return 768
            return 384

        def __init__(self, model_name: str):
            created_models.append(model_name)

    monkeypatch.setattr(
        "fastembed.TextEmbedding",
        FakeTextEmbedding,
    )
    module_path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "services"
        / "embedder.py"
    )

    return runpy.run_path(str(module_path)), created_models


def test_embedder_rejects_model_with_wrong_dimension(monkeypatch):
    module, created_models = load_embedder_module(monkeypatch)

    with pytest.raises(
        module["EmbeddingDimensionError"],
        match="produces 768-dimensional vectors.*expects 384",
    ):
        module["EmbedderService"]("wrong-dimension-model")

    assert "wrong-dimension-model" not in created_models
