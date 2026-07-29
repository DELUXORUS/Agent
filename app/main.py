import asyncio
import logging
from fastapi import FastAPI
from app.core.broker import broker
from contextlib import asynccontextmanager
from app.api.router import router as api_router

logger = logging.getLogger("uvicorn")

@asynccontextmanager
async def lifespan(app: FastAPI):
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to RabbitMQ (attempt {attempt}/{max_retries})...")
            await broker.start()
            logger.info("Successfully connected to RabbitMQ!")
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error("Could not connect to RabbitMQ. Exiting.")
                raise e
            logger.warning(f"RabbitMQ not ready yet ({e}). Retrying in 2 seconds...")
            await asyncio.sleep(2)

    yield
    await broker.stop()
app = FastAPI(lifespan=lifespan)
app.include_router(api_router,
                   prefix="/api/v1",)

@app.get("/health")
async def health_check():
    return {"status": "ok"}