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
