import asyncio
from sqlalchemy import text
from app.db.database import engine, async_session_maker

async def clear_database():
    print("🧹 Очистка таблиц базы данных...")
    async with async_session_maker() as session:
        # TRUNCATE мгновенно сбрасывает данные и обнуляет счетчики ID (RESTART IDENTITY)
        # CASCADE удаляет связанные записи из user_movies_history (если есть FK)
        await session.execute(
            text("TRUNCATE TABLE movies, user_movies_history RESTART IDENTITY CASCADE;")
        )
        await session.commit()
    print("✅ Все данные из таблиц удалены!")

if __name__ == "__main__":
    asyncio.run(clear_database())