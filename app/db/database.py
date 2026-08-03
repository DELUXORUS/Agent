from sqlalchemy import text
from app.db.models import Base, Movie, UserMovieHistory
from app.config import settings
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)

engine = create_async_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
    echo=True
)

async_session_maker = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db():
    async with async_session_maker() as session:
        yield session