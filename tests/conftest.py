import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Set test configuration before application modules are imported.
os.environ.update({
    "BOT_TOKEN": "test-token",
    "OPENROUTER_API_KEY": "test-key",
    "RABBITMQ_URL": "amqp://guest:guest@localhost:5672/",
    "EMBEDDER_MODEL_NAME": "test-embedder",
    "LLM_MODEL_NAME": "test-llm",
    "OPENROUTER_URL": "http://localhost:1",
    "PROMPTS_PATH": str(Path(__file__).resolve().parents[1] / "app/agent/system_prompts.yaml"),
})

import pytest

from app.schemas import MovieDTO


@pytest.fixture
def mock_llm():
    llm = MagicMock()

    mock_response = MagicMock()
    mock_response.content = "Тестовый ответ от мок-LLM"
    llm.ainvoke = AsyncMock(return_value=mock_response)

    structured_mock = MagicMock()
    structured_mock.ainvoke = AsyncMock()
    llm.with_structured_output = MagicMock(return_value=structured_mock)

    return llm


@pytest.fixture
def sample_state():
    return {
        "request_id": "test-request",
        "user_query": "Посоветуй фантастику про космос",
        "user_id": 123456789,
        "candidates": [],
        "final_response": "",
    }


@pytest.fixture
def sample_movie_dto():
    return MovieDTO(
        id=1,
        title="Interstellar",
        overview="A team of explorers travel through a wormhole in space.",
        genres="Adventure, Drama, Sci-Fi",
        release_date="2014-11-05",
        vote_average=8.4,
    )
