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

    # Заполняем NaN пустой строкой для текстовых полей
    df["title"] = df["title"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["cast"] = df["cast"].fillna("")
    df["tagline"] = df["tagline"].fillna("")

    print(f"CSV прочитан за {time.perf_counter() - t0:.2f} сек.")

    # 2. Собираем единый контекстный текст для эмбеддинга
    texts = (
            "Title: " + df["title"] +
            ". Tagline: " + df["tagline"] +
            ". Genres: " + df["genres"] +
            ". Cast: " + df["cast"] +
            ". Overview: " + df["overview"]
    ).tolist()

    # 3. Генерация векторов (с поддержкой async/await)
    print("\nГенерация эмбеддингов...")
    t_emb_start = time.perf_counter()

    EMBEDDING_BATCH_SIZE = 1024
    embeddings = []

    with tqdm(total=len(texts), desc="Векторизация", unit="текст") as pbar:
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i: i + EMBEDDING_BATCH_SIZE]
            # Добавили await!
            batch_vecs = await embedder.get_embeddings(batch_texts)
            embeddings.extend(batch_vecs)
            pbar.update(len(batch_texts))

    emb_time = time.perf_counter() - t_emb_start
    print(
        f"Векторизация {len(texts)} текстов завершена за {emb_time:.2f} сек! ({len(texts) / emb_time:.1f} текстов/сек)"
    )

    # 4. Подготовка данных для asyncpg (с учетом cast и tagline)
    print("\nПодготовка данных для быстрой записи...")
    records = []

    # Конвертируем release_date аккуратно
    release_dates = pd.to_datetime(df["release_date"], errors="coerce").dt.date

    for row, rel_date, vec in zip(df.itertuples(), release_dates, embeddings):
        records.append((
            row.title,
            row.overview if row.overview else None,
            row.genres if row.genres else None,
            row.cast if row.cast else None,  # <-- Добавили cast
            row.tagline if row.tagline else None,  # <-- Добавили tagline
            rel_date if pd.notnull(rel_date) else None,
            float(row.vote_average) if pd.notnull(getattr(row, 'vote_average', None)) else None,
            str(vec)  # Преобразуем вектор [0.1, ...] в строку формата pgvector
        ))

    # 5. Высокоскоростной batch INSERT через драйвер asyncpg
    print("\nЗапись в PostgreSQL (через asyncpg executemany)...")
    t_db_start = time.perf_counter()

    raw_conn = await engine.raw_connection()
    try:
        asyncpg_conn = raw_conn.driver_connection

        # Включаем колонки cast и tagline в запрос
        insert_query = """
                       INSERT INTO movies (title, overview, genres, cast, tagline, release_date, vote_average, \
                                           embedding)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                       """

        DB_BATCH_SIZE = 5000
        with tqdm(total=len(records), desc="  Вставка в БД", unit="запись") as pbar:
            for i in range(0, len(records), DB_BATCH_SIZE):
                batch = records[i: i + DB_BATCH_SIZE]
                await asyncpg_conn.executemany(insert_query, batch)
                pbar.update(len(batch))

    finally:
        await raw_conn.close()

    db_time = time.perf_counter() - t_db_start
    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 50)
    print(f"40 000 фильмов успешно записаны за {total_time:.2f} сек!")
    print(f"Векторизация: {emb_time:.2f} сек")
    print(f"Запись в БД:   {db_time:.2f} сек")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(seed_movies())