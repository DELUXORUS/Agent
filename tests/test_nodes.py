from unittest.mock import AsyncMock, patch
import pytest

from app.agent.nodes import (
    intent_classification_node,
    general_chat,
    synthesis_response_node,
    search_for_guess_node,
)
from app.agent.state import IntentClassification, MovieFilter


@pytest.mark.asyncio
async def test_intent_classification_node_success(sample_state, mock_llm):
    """Тест успешной классификации интента."""
    # Настраиваем возвращаемое значение для structured_llm.ainvoke
    mock_structured = mock_llm.with_structured_output.return_value
    mock_structured.ainvoke.return_value = IntentClassification(intent="recommend_movies")

    result = await intent_classification_node(sample_state, mock_llm)

    assert result["intent"] == "recommend_movies"
    mock_llm.with_structured_output.assert_called_once()


@pytest.mark.asyncio
async def test_intent_classification_fallback_on_error(sample_state, mock_llm):
    """Тест фолбэка при ошибке LLM."""
    mock_structured = mock_llm.with_structured_output.return_value
    mock_structured.ainvoke.side_effect = Exception("API Timeout")

    result = await intent_classification_node(sample_state, mock_llm)

    # При ошибке должен вернуть 'general_chat'
    assert result["intent"] == "general_chat"


@pytest.mark.asyncio
async def test_general_chat_node(sample_state, mock_llm):
    """Тест обычного чата."""
    result = await general_chat(sample_state, mock_llm)

    assert result["final_response"] == "Тестовый ответ от мок-LLM"
    mock_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_search_for_guess_node_insufficient_info(sample_state, mock_llm):
    """Тест guardrail проверки на слишком короткий запрос."""
    # Симулируем короткий парсинг
    mock_structured = mock_llm.with_structured_output.return_value
    mock_structured.ainvoke.return_value = MovieFilter(
        query_text="a",  # Слишком короткий запрос (< 3 символов)
        is_semantic_search_needed=True
    )

    result = await search_for_guess_node(sample_state, mock_llm)

    assert result["error_reason"] == "insufficient_information"
    assert result["found_movies"] == []


@pytest.mark.asyncio
async def test_synthesis_response_node_insufficient_info(sample_state, mock_llm):
    """Тест реакции синтезатора на недостаток информации."""
    sample_state["error_reason"] = "insufficient_information"

    result = await synthesis_response_node(sample_state, mock_llm)

    assert "слишком мало деталей" in result["final_response"]
    # LLM не должна вызываться, если сработал guardrail
    mock_llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_synthesis_response_node_with_movies(sample_state, mock_llm, sample_movie_dto):
    """Тест синтеза ответа, когда фильмы найдены."""
    sample_state["found_movies"] = [sample_movie_dto]
    sample_state["intent"] = "guess_movie"

    result = await synthesis_response_node(sample_state, mock_llm)

    assert result["final_response"] == "Тестовый ответ от мок-LLM"
    mock_llm.ainvoke.assert_called_once()