from fastapi import APIRouter, Request, Response, status

from app.core.broker import broker
from app.schemas import TelegramMessageTask

router = APIRouter(prefix="/webhook",
                   tags=["Telegram Webhook"])


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

    return Response(status_code=status.HTTP_200_OK)
