from app.config import Settings


def test_llm_fallback_models_are_normalized_and_deduplicated():
    settings = Settings(
        LLM_MODEL_NAME="primary-model",
        LLM_FALLBACK_MODELS=(
            " fallback-a, primary-model, fallback-a, , fallback-b "
        ),
    )

    assert settings.llm_fallback_models == [
        "fallback-a",
        "fallback-b",
    ]
