import runpy
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserMovieHistory
from tests.factories import make_movie_dto


def load_handler(monkeypatch, graph, session_factory):
    graph_module = ModuleType("app.agent.graph")
    graph_module.graph = graph
    monkeypatch.setitem(sys.modules, "app.agent.graph", graph_module)

    database_module = ModuleType("app.db.database")
    database_module.async_session_maker = session_factory
    monkeypatch.setitem(sys.modules, "app.db.database", database_module)

    handler_path = Path(__file__).resolve().parents[2] / "app/handler/handler.py"
    namespace = runpy.run_path(str(handler_path))
    return SimpleNamespace(**namespace)


@pytest.mark.asyncio
async def test_generate_response_builds_agent_state_and_returns_selected_movies(
    monkeypatch,
):
    movies = [make_movie_dto(id=3, title="Moon station")]
    graph = Mock(ainvoke=AsyncMock(return_value={
        "final_response": "Вот подходящий фильм",
        "selected_movies": movies,
    }))
    handler = load_handler(monkeypatch, graph, Mock())

    response, selected_movies = await handler.generate_response(
        text="Посоветуй фильм про космос",
        username="tester",
        user_id=42,
    )

    assert response == "Вот подходящий фильм"
    assert selected_movies == movies
    state = graph.ainvoke.await_args.args[0]
    assert state["user_id"] == 42
    assert state["user_query"] == "Посоветуй фильм про космос"
    assert set(state) == {"request_id", "user_id", "user_query"}
    UUID(state["request_id"])


@pytest.mark.asyncio
async def test_message_handler_builds_keyboard_and_sends_response(monkeypatch):
    graph = Mock()
    handler = load_handler(monkeypatch, graph, Mock())
    movies = [make_movie_dto(id=3, title="Moon station")]
    message_handler = handler.handle_telegram_messages._original_call
    message_handler.__globals__["generate_response"] = AsyncMock(
        return_value=("Вот подходящий фильм", movies)
    )
    telegram = Mock(
        make_movies_keyboard=Mock(return_value={
            "inline_keyboard": [[{
                "text": "Смотрел №1",
                "callback_data": "watch:3:1",
            }]],
        }),
        send_message=AsyncMock(),
    )
    message_handler.__globals__["telegram_service"] = telegram

    await message_handler({
        "text": "Посоветуй фильм",
        "user_id": 42,
        "username": "tester",
        "chat_id": 500,
    })

    telegram.make_movies_keyboard.assert_called_once_with(movies)
    telegram.send_message.assert_awaited_once_with(
        500,
        "Вот подходящий фильм",
        reply_markup=telegram.make_movies_keyboard.return_value,
    )


@pytest.mark.asyncio
async def test_watch_callback_writes_history_and_updates_keyboard(
    monkeypatch,
    pg_catalog: AsyncSession,
):
    @asynccontextmanager
    async def session_factory():
        yield pg_catalog

    handler = load_handler(monkeypatch, Mock(), session_factory)
    telegram = Mock(
        answer_callback_query=AsyncMock(),
        edit_reply_markup=AsyncMock(),
    )
    callback_handler = handler.handle_telegram_callbacks._original_call
    callback_handler.__globals__["telegram_service"] = telegram
    callback_data = {
        "callback_query_id": "callback-1",
        "callback_data": "watch:3:2",
        "user_id": 777,
        "chat_id": 500,
        "message_id": 600,
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "Смотрел №1", "callback_data": "watch:2:1"},
                {"text": "Смотрел №2", "callback_data": "watch:3:2"},
            ]],
        },
    }

    await callback_handler(callback_data)

    history = (await pg_catalog.execute(
        select(UserMovieHistory)
        .where(
            UserMovieHistory.user_id == 777,
            UserMovieHistory.movie_id == 3,
        )
    )).scalar_one()
    assert history.user_id == 777
    telegram.answer_callback_query.assert_awaited_once_with(
        "callback-1",
        "Фильм №2 добавлен в просмотренные!",
    )
    telegram.edit_reply_markup.assert_awaited_once_with(
        chat_id=500,
        message_id=600,
        reply_markup={
            "inline_keyboard": [[
                {"text": "Смотрел №1", "callback_data": "watch:2:1"},
                {"text": "№2", "callback_data": "ignore"},
            ]],
        },
    )
