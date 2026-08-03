from unittest.mock import AsyncMock, MagicMock
import pytest
from langchain_core.messages import HumanMessage
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
        "messages": [HumanMessage(content="Посоветуй фантастику про космос")],
        "user_id": 123456789,
        "intent": None,
        "parsed_filter": None,
        "found_movies": [],
        "final_response": "",
        "error_reason": None,
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