import time
import asyncio
import pandas as pd
from tqdm import tqdm

from app.db.operations import Operations
from app.db.database import init_db, async_session_maker, engine
from scripts.local_embedder import embedder


async def seed_movies():
    start_time = time.perf_counter()
    print("Инициализация БД (создание таблиц)...")
    await init_db()

    async with async_session_maker() as session:
        operation = Operations(session)
        count = await operation.get_movies_count()
        if count > 0:
            print(f"База данных уже содержит {count} фильмов. Пропуск забивки.")
            return

    print("Чтение и обработка CSV...")
    t0 = time.perf_counter()
    df = pd.read_csv('data/movies.csv')

    # 🎯 1. Приводим типы для качественной фильтрации
    df["vote_count"] = pd.to_numeric(df.get("vote_count", 0), errors="coerce").fillna(0)
    df["vote_average"] = pd.to_numeric(df.get("vote_average", 0), errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0)

    # 🎯 2. Жесткий отсев "мусора" (отсутствие описания / каста / слишком мало голосов)
    # Фильтруем фильмы, у которых менее 50 голосов и короткое описание (< 20 символов)
    df["overview_clean"] = df["overview"].fillna("").astype(str)
    df = df[
        (df["vote_count"] >= 50) &
        (df["overview_clean"].str.len() > 20)
    ]

    # 🎯 3. Сортировка: сначала самые популярные и с наибольшим числом оценок
    df = df.sort_values(
        by=["vote_count", "popularity", "vote_average"],
        ascending=[False, False, False]
    ).head(40000)

    df["title"] = df["title"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["keywords"] = df["keywords"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["credits"] = df["credits"].fillna("")
    df["tagline"] = df["tagline"].fillna("")

    print(f"CSV прочитан и отфильтрован за {time.perf_counter() - t0:.2f} сек. Отобрано топ-{len(df)} фильмов!")

    texts = (
            "Title: " + df["title"] +
            ". Tagline: " + df["tagline"] +
            ". Genres: " + df["genres"] +
            ". Keywords: " + df["keywords"] +
            ". Credits: " + df["credits"] +
            ". Overview: " + df["overview"]
    ).tolist()

    print("\nГенерация эмбеддингов...")
    t_emb_start = time.perf_counter()

    embeddings = await embedder.get_embeddings(texts, batch_size=512)
    # EMBEDDING_BATCH_SIZE = 1024
    # embeddings = []

    # with tqdm(total=len(texts), desc="Векторизация", unit="текст") as pbar:
    #     for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
    #         batch_texts = texts[i: i + EMBEDDING_BATCH_SIZE]
    #         batch_vecs = await embedder.get_embeddings(batch_texts)
    #         embeddings.extend(batch_vecs)
    #         pbar.update(len(batch_texts))

    emb_time = time.perf_counter() - t_emb_start
    print(
        f"Векторизация {len(texts)} текстов завершена за {emb_time:.2f} сек! ({len(texts) / emb_time:.1f} текстов/сек)"
    )

    print("\nПодготовка данных для быстрой записи...")
    records = []

    # Предварительно конвертируем типы
    release_dates = pd.to_datetime(df["release_date"], errors="coerce").dt.date
    vote_averages = pd.to_numeric(df["vote_average"], errors="coerce")

    for row, rel_date, vote_avg, vec in zip(df.itertuples(), release_dates, vote_averages, embeddings):
        records.append((
            row.title,                                           # $1: title
            row.overview if row.overview else None,              # $2: overview
            row.genres if row.genres else None,                  # $3: genres
            row.credits if row.credits else None,                # $4: credits
            row.tagline if row.tagline else None,                # $5: tagline
            rel_date if pd.notnull(rel_date) else None,          # $6: release_date
            float(vote_avg) if pd.notnull(vote_avg) else None,   # $7: vote_average
            row.keywords if row.keywords else None,              # $8: keywords
            str(vec)                                             # $9: embedding
        ))

    # 5. Высокоскоростной batch INSERT через драйвер asyncpg
    print("\nЗапись в PostgreSQL (через asyncpg executemany)...")
    t_db_start = time.perf_counter()

    raw_conn = await engine.raw_connection()
    try:
        asyncpg_conn = raw_conn.driver_connection

        insert_query = """
                       INSERT INTO movies (title, overview, genres, credits, tagline, release_date, vote_average, keywords, embedding)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::vector)
                       """

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
    print(f"{len(records)} качественных фильмов успешно записаны за {total_time:.2f} сек!")
    print(f"Векторизация: {emb_time:.2f} сек")
    print(f"Запись в БД:   {db_time:.2f} сек")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(seed_movies())