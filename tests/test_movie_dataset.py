import pandas as pd
import pytest

from app.core.text_normalization import normalize_search_text
from scripts.movie_dataset import (
    extract_directors,
    extract_names,
    prepare_and_filter_movies,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Christopher   Nolan  ", "christopher nolan"),
        ("Science Fiction", "science fiction"),
        ("Straße", "strasse"),
        ("Ｔｈｅ　Ｇａｍｅ", "the game"),
        ("  \t\n  ", ""),
    ],
)
def test_normalize_search_text(value, expected):
    assert normalize_search_text(value) == expected


def test_normalize_search_text_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        normalize_search_text(42)


def test_extract_names_normalizes_and_removes_duplicates():
    value = str([
        {"name": " Science   Fiction "},
        {"name": "science fiction"},
        {"name": "Drama"},
    ])

    assert extract_names(value) == ("science fiction", "drama")


def test_extract_directors_normalizes_names():
    value = str([
        {"job": "Director", "name": " Christopher   Nolan "},
        {"job": "Producer", "name": "Emma Thomas"},
    ])

    assert extract_directors(value) == ("christopher nolan",)


def test_prepare_movies_preserves_titles_and_adds_normalized_titles():
    dataframe = pd.DataFrame([{
        "id": 2649,
        "imdb_id": "tt0119174",
        "title": " Ｔｈｅ   Game ",
        "original_title": "The Game",
        "original_language": "en",
        "overview": "A" * 80,
        "tagline": None,
        "genres": str([{"name": " Drama "}, {"name": "Thriller"}]),
        "release_date": "1997-09-12",
        "runtime": 129,
        "vote_average": 7.5,
        "vote_count": 1000,
        "adult": False,
        "video": False,
        "status": "Released",
        "actors": ("michael douglas",),
        "directors": ("david fincher",),
        "keywords": ("game",),
    }])

    prepared, report = prepare_and_filter_movies(dataframe)
    movie = prepared.iloc[0]

    assert movie["title"] == "Ｔｈｅ   Game"
    assert movie["normalized_title"] == "the game"
    assert movie["original_title"] == "The Game"
    assert movie["normalized_original_title"] == "the game"
    assert movie["genres"] == ("drama", "thriller")
    assert report["output_rows"] == 1


@pytest.fixture
def seed_record():
    from datetime import date
    from scripts.movie_dataset import MovieSeedRecord

    return MovieSeedRecord(
        tmdb_id=2649, imdb_id="tt0119174", title="The Game",
        normalized_title="the game", original_title="  THE   GAME ",
        normalized_original_title="the game", original_language="en",
        overview="A banker receives an unusual birthday gift.",
        tagline=None, genres=("drama", "thriller"),
        actors=("michael douglas",), directors=("david fincher",),
        keywords=(), release_date=date(1997, 9, 12),
        runtime=129, vote_average=7.5, vote_count=1000,
    )


def test_embedding_text_preserves_display_text_and_omits_duplicate_title(seed_record):
    from scripts.movie_dataset import build_movie_embedding_text

    assert build_movie_embedding_text(seed_record) == (
        "Title: The Game\n"
        "Overview: A banker receives an unusual birthday gift.\n"
        "Genres: drama, thriller"
    )


def test_embedding_text_includes_original_title_and_limits_keywords(seed_record):
    from dataclasses import replace
    from scripts.movie_dataset import build_movie_embedding_text

    movie = replace(seed_record, original_title="Игра", tagline="An unusual gift.",
                    keywords=tuple(f"keyword-{i}" for i in range(30)))
    text = build_movie_embedding_text(movie)
    assert "Original title: Игра" in text
    assert "Tagline: An unusual gift." in text
    assert text.splitlines()[-1] == "Keywords: " + ", ".join(movie.keywords[:25])
    assert len(movie.keywords) == 30
    assert text.index("Overview:") < text.index("Keywords:")


def test_embedding_text_handles_missing_optional_metadata(seed_record):
    from dataclasses import replace
    from scripts.movie_dataset import build_movie_embedding_text

    movie = replace(seed_record, original_title=None, genres=())
    assert build_movie_embedding_text(movie).splitlines() == [
        "Title: The Game", "Overview: A banker receives an unusual birthday gift.",
    ]


def test_movie_rows_preserve_pairing_and_python_types(seed_record):
    from dataclasses import replace
    from app.constants import EMBEDDING_DIMENSION
    from scripts.seed_db import build_movie_rows

    second = replace(seed_record, tmdb_id=2650, title="Another film")
    vectors = [[0.1] * EMBEDDING_DIMENSION, [0.2] * EMBEDDING_DIMENSION]
    rows = build_movie_rows([seed_record, second], vectors)
    assert [row["tmdb_id"] for row in rows] == [2649, 2650]
    assert [row["embedding"] for row in rows] == vectors
    assert rows[0]["release_date"] == seed_record.release_date
    assert rows[0]["tagline"] is None
    assert rows[0]["title"] == seed_record.title
    assert "id" not in rows[0]
    for field in ("genres", "actors", "directors", "keywords"):
        assert rows[0][field] == list(getattr(seed_record, field))
    rows[0]["genres"].append("comedy")
    rows[0]["embedding"][0] = 1.0
    assert "comedy" not in seed_record.genres
    assert "comedy" not in rows[1]["genres"]
    assert vectors[0][0] == 0.1


@pytest.mark.parametrize("movie_count,vector_count", [(1, 0), (0, 1), (1, 2), (2, 1)])
def test_movie_rows_reject_mismatched_counts(seed_record, movie_count, vector_count):
    from scripts.seed_db import build_movie_rows

    with pytest.raises(ValueError):
        build_movie_rows([seed_record] * movie_count, [[0.1]] * vector_count)


def test_movie_rows_accept_empty_batch():
    from scripts.seed_db import build_movie_rows

    assert build_movie_rows([], []) == []


@pytest.mark.asyncio
async def test_seed_movies_processes_records_in_batches(monkeypatch, seed_record):
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from app.constants import EMBEDDING_DIMENSION
    import scripts.seed_db as seed_db

    movies = [
        replace(seed_record, tmdb_id=seed_record.tmdb_id + index)
        for index in range(5)
    ]

    async def embed(texts, batch_size):
        assert batch_size == 32
        return [[0.1] * EMBEDDING_DIMENSION for _ in texts]

    embedder = SimpleNamespace(get_embeddings=AsyncMock(side_effect=embed))
    upsert = AsyncMock()
    monkeypatch.setattr(seed_db, "upsert_movies", upsert)

    seeded_count = await seed_db.seed_movies(
        AsyncMock(),
        movies,
        embedder,
        batch_size=2,
    )

    assert seeded_count == 5
    assert embedder.get_embeddings.await_count == 3
    assert upsert.await_count == 3
    assert [
        [row["tmdb_id"] for row in call.args[1]]
        for call in upsert.await_args_list
    ] == [[2649, 2650], [2651, 2652], [2653]]


@pytest.mark.asyncio
async def test_seed_movies_rejects_non_positive_batch_size(seed_record):
    from unittest.mock import AsyncMock

    from scripts.seed_db import seed_movies

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        await seed_movies(
            AsyncMock(),
            [seed_record],
            AsyncMock(),
            batch_size=0,
        )
