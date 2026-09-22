import runpy
import warnings
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
            if model_name == "pooling-warning-model":
                warnings.warn(
                    "The model pooling-warning-model now uses mean pooling "
                    "instead of CLS embedding. In order to preserve the "
                    "previous behaviour, pin an older version.",
                    UserWarning,
                    stacklevel=2,
                )
            elif model_name == "other-warning-model":
                warnings.warn(
                    "Unrelated model warning",
                    UserWarning,
                    stacklevel=2,
                )
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


def test_embedder_suppresses_known_pooling_migration_warning(monkeypatch):
    module, _ = load_embedder_module(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module["EmbedderService"]("pooling-warning-model")

    assert caught == []


def test_embedder_keeps_unrelated_warnings_visible(monkeypatch):
    module, _ = load_embedder_module(monkeypatch)

    with pytest.warns(UserWarning, match="Unrelated model warning"):
        module["EmbedderService"]("other-warning-model")
