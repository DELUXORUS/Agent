from unittest.mock import AsyncMock

import pytest

from app.api import router as router_module


@pytest.mark.asyncio
async def test_message_webhook_publishes_validated_task(monkeypatch):
    publish = AsyncMock()
    monkeypatch.setattr(router_module.broker, "publish", publish)

    response = await router_module.telegram_webhook({
        "message": {
            "message_id": 10,
            "from": {"id": 42, "username": "tester"},
            "chat": {"id": 500},
            "text": "Посоветуй фильм",
        },
    })

    assert response.status_code == 200
    task = publish.await_args.args[0]
    assert task.model_dump() == {
        "user_id": 42,
        "username": "tester",
        "chat_id": 500,
        "message_id": 10,
        "text": "Посоветуй фильм",
    }
    assert publish.await_args.kwargs == {"queue": "telegram_messages"}


@pytest.mark.asyncio
async def test_callback_webhook_preserves_keyboard(monkeypatch):
    publish = AsyncMock()
    monkeypatch.setattr(router_module.broker, "publish", publish)
    keyboard = {
        "inline_keyboard": [[{
            "text": "Смотрел №1",
            "callback_data": "watch:3:1",
        }]],
    }

    response = await router_module.telegram_webhook({
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 42},
            "data": "watch:3:1",
            "message": {
                "message_id": 10,
                "chat": {"id": 500},
                "reply_markup": keyboard,
            },
        },
    })

    assert response.status_code == 200
    task = publish.await_args.args[0]
    assert task.model_dump() == {
        "user_id": 42,
        "chat_id": 500,
        "message_id": 10,
        "callback_query_id": "callback-1",
        "callback_data": "watch:3:1",
        "reply_markup": keyboard,
    }
    assert publish.await_args.kwargs == {"queue": "telegram_callbacks"}
