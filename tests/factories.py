from datetime import date

from app.schemas import MovieDTO


def make_movie_dto(**overrides) -> MovieDTO:
    values = {
        "id": 1,
        "tmdb_id": 1001,
        "title": "Example",
        "overview": "An example movie overview.",
        "genres": ["drama"],
        "release_date": date(2000, 1, 1),
        "runtime": 100,
        "vote_average": 7.0,
        "vote_count": 100,
    }
    values.update(overrides)
    return MovieDTO(**values)
