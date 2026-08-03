from unittest.mock import AsyncMock, patch
import pytest

from app.agent.nodes import (
    general_chat,
    synthesis_response_node,
    search_for_guess_node,
)
from app.agent.state import MovieFilter



@pytest.mark.asyncio
async def test_general_chat_node(sample_state, mock_llm):
    result = await general_chat(sample_state, mock_llm)

    assert result["final_response"] == "Тестовый ответ от мок-LLM"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_search_for_guess_node_insufficient_info(sample_state, mock_llm):
    mock_structured = mock_llm.with_structured_output.return_value
    mock_structured.ainvoke.return_value = MovieFilter(
        is_semantic_search_needed=True
    )

    result = await search_for_guess_node(sample_state, mock_llm)

    assert result["error_reason"] == "insufficient_information"
    assert result["found_movies"] == []


@pytest.mark.asyncio
async def test_synthesis_response_node_insufficient_info(sample_state, mock_llm):
    sample_state["error_reason"] = "insufficient_information"

    result = await synthesis_response_node(sample_state, mock_llm)

    assert "слишком мало деталей" in result["final_response"]
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_synthesis_response_node_with_movies(sample_state, mock_llm, sample_movie_dto):
    sample_state["found_movies"] = [sample_movie_dto]
    sample_state["intent"] = "guess_movie"

    result = await synthesis_response_node(sample_state, mock_llm)

    assert result["final_response"] == "Тестовый ответ от мок-LLM"
    mock_llm.ainvoke.assert_called_once()