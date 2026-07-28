import asyncio
import time
import pandas as pd
from tqdm import tqdm

from app.db.operations import Operations
from app.db.database import init_db, async_session_maker, engine
from app.service.embedder import embedder


async def seed_movies():
    start_time = time.perf_counter()
    print("Инициализация БД (создание таблиц)...")
    await init_db()

    # 1. Проверяем наличие записей
    async with async_session_maker() as session:
        operation = Operations(session)
        count = await operation.get_movies_count()
        if count > 0:
            print(f"База данных уже содержит {count} фильмов. Пропуск забивки.")
            return

    print("Чтение и обработка CSV...")
    t0 = time.perf_counter()
    df = pd.read_csv('data/movies.csv')
    df = df.sort_values(by="popularity", ascending=False).head(40000)
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.date
    df = df.fillna('')
    print(f"  └─ CSV прочитан за {time.perf_counter() - t0:.2f} сек.")

    # 2. Формируем тексты
    texts = (df['title'] + ": " + df['overview']).tolist()

    # 3. Генерация векторов на GPU
    print("\nГенерация эмбеддингов на GPU...")
    t_emb_start = time.perf_counter()

    EMBEDDING_BATCH_SIZE = 1024
    embeddings = []

    with tqdm(total=len(texts), desc="  Векторизация", unit="текст") as pbar:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i: i + EMBEDDING_BATCH_SIZE]
            batch_vecs = embedder.get_embeddings(batch_texts)
            embeddings.extend(batch_vecs)
            pbar.update(len(batch_texts))

    emb_time = time.perf_counter() - t_emb_start
    print(
        f"  └─ Векторизация {len(texts)} текстов завершена за {emb_time:.2f} сек! ({len(texts) / emb_time:.1f} текстов/сек)")

    # 4. Подготовка данных для asyncpg
    print("\nПодготовка данных для быстрой записи...")
    records = []
    for row, vec in zip(df.itertuples(), embeddings):
        records.append((
            row.title,
            row.overview if row.overview else None,
            row.genres if row.genres else None,
            row.release_date if row.release_date else None,
            getattr(row, 'vote_average', None) if getattr(row, 'vote_average', None) != '' else None,
            str(vec)  # Преобразуем вектор [0.1, ...] в строку
        ))

    # 5. Высокоскоростной batch INSERT через драйвер asyncpg
    print("\nЗапись в PostgreSQL (через asyncpg executemany)...")
    t_db_start = time.perf_counter()

    raw_conn = await engine.raw_connection()
    try:
        # Извлекаем оригинальное соединение asyncpg
        asyncpg_conn = raw_conn.driver_connection

        insert_query = """
                       INSERT INTO movies (title, overview, genres, release_date, vote_average, embedding)
                       VALUES ($1, $2, $3, $4, $5, $6::vector) \
                       """

        # Вставляем крупными порциями по 5000 записей
        DB_BATCH_SIZE = 5000
        with tqdm(total=len(records), desc="  Вставка в БД", unit="запись") as pbar:
            for i in range(0, len(records), DB_BATCH_SIZE):
                batch = records[i: i + DB_BATCH_SIZE]
                await asyncpg_conn.executemany(insert_query, batch)
                pbar.update(len(batch))

    finally:
        raw_conn.close()

    db_time = time.perf_counter() - t_db_start
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 50)
    print(f"⚡ 40 000 фильмов успешно забиты за {total_time:.2f} сек!")
    print(f"  • Векторизация: {emb_time:.2f} сек")
    print(f"  • Запись в БД:   {db_time:.2f} сек")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(seed_movies())