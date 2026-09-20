from datetime import date

import pytest

from app.core.text_normalization import normalize_search_text
from app.db.models import Movie
from app.db.operations import Operations


@pytest.mark.asyncio
async def test_normalized_titles_original_names_and_years(pg_session):
    for movie_id, title, original, released in [
        (1, "Spirited Away", "千と千尋の神隠し", date(2001, 1, 1)),
        (2, "Spirited Away", "Spirited Away", date(2020, 12, 31)),
        (3, "Other title", None, date(2001, 6, 1)),
    ]:
        pg_session.add(Movie(
            id=movie_id, tmdb_id=movie_id, title=title,
            normalized_title=normalize_search_text(title),
            original_title=original,
            normalized_original_title=(
                normalize_search_text(original) if original else None
            ),
            overview="An example movie.", genres=["fantasy"],
            release_date=released, runtime=120, vote_average=8, vote_count=100,
        ))
    await pg_session.flush()
    operations = Operations(pg_session)
    for query, year, expected in [
        ("  SPIRITED\t  AWAY  ", None, [1, 2]),
        ("Ｓｐｉｒｉｔｅｄ　Ａｗａｙ", None, [1, 2]),
        ("千と千尋の神隠し", None, [1]),
        ("spirited away", 2001, [1]),
        ("spirited away", 2020, [2]),
        ("千と千尋の神隠し", 2020, []),
        ("other title", None, [3]),
        ("spirited", None, []),
        ("%", None, []),
        ("missing", None, []),
    ]:
        movies = await operations.find_movies_by_title(query, year)
        assert [movie.id for movie in movies] == expected, (query, year)
