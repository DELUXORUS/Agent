from fastapi import APIRouter, Request, Response, status

from app.core.broker import broker
from app.schemas import TelegramMessageTask

router = APIRouter(prefix="/webhook",
                   tags=["Telegram Webhook"])


@router.post("/telegram")
async def telegram_webhook(data: dict):
    # data = await request.json()

    if "message" in data and "text" in data["message"]:
        msg = data["message"]

        task = TelegramMessageTask(
            user_id=msg["from"]["id"],
            chat_id=msg["chat"]["id"],
            message_id=msg["message_id"],
            prompt=msg["text"],
        )

        await broker.publish(task, queue="telegram_tasks")

    return Response(status_code=status.HTTP_200_OK)
