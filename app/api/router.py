from fastapi import APIRouter, Response, status

from app.core.broker import broker
from app.schemas import TelegramCallbackTask, TelegramMessageTask

router = APIRouter(prefix="/webhook", tags=["Telegram Webhook"])


@router.post("/telegram")
async def telegram_webhook(data: dict):
    if "message" in data and "text" in data["message"]:
        msg = data["message"]

        username = (
            msg["from"].get("username")
            or msg["from"].get("first_name")
            or "User"
        )

        task = TelegramMessageTask(
            user_id=msg["from"]["id"],
            username=username,
            chat_id=msg["chat"]["id"],
            message_id=msg["message_id"],
            text=msg["text"],
        )

        await broker.publish(task, queue="telegram_messages")

    elif "callback_query" in data:
        cb = data["callback_query"]
        msg = cb.get("message", {})

        callback_task = TelegramCallbackTask(
            user_id=cb["from"]["id"],
            chat_id=msg.get("chat", {}).get("id", 0),
            message_id=msg.get("message_id", 0),
            callback_query_id=cb["id"],
            callback_data=cb.get("data", ""),
            reply_markup=msg.get("reply_markup", {}),
        )
        await broker.publish(callback_task, queue="telegram_callbacks")

    return Response(status_code=status.HTTP_200_OK)