from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas import MovieDTO
from app.db.models import Movie
from tests.factories import make_movie_dto


def test_movie_dto_validation(sample_movie_dto):
    assert sample_movie_dto.title == "Interstellar"
    assert sample_movie_dto.vote_average == 8.4


def test_movie_dto_reads_orm_and_serializes_date_without_storage_fields():
    movie = Movie(
        id=1, tmdb_id=1001, imdb_id=None, title="Interstellar",
        normalized_title="interstellar", original_title=None,
        normalized_original_title=None, original_language=None,
        overview="Explorers travel through space.", tagline=None,
        release_date=date(2014, 11, 7), runtime=169,
        vote_average=8.4, vote_count=100,
        genres=["science fiction"], actors=[], directors=[], keywords=[],
        embedding=[0.1] * 384,
    )
    dto = MovieDTO.model_validate(movie)
    assert dto.release_date == date(2014, 11, 7)
    assert dto.genres == ["science fiction"]
    assert dto.actors == dto.directors == dto.keywords == []
    assert dto.imdb_id is dto.original_title is dto.original_language is dto.tagline is None
    payload = dto.model_dump(mode="json")
    assert payload["release_date"] == "2014-11-07"
    assert not {"embedding", "normalized_title", "normalized_original_title"} & payload.keys()


@pytest.mark.parametrize("values", [
    {"runtime": 0}, {"runtime": -1},
    {"vote_average": -0.1}, {"vote_average": 10.1},
    {"vote_average": float("nan")}, {"vote_average": float("inf")},
    {"vote_count": -1}, {"release_date": None},
    {"release_date": "not-a-date"}, {"vote_average": None},
    {"genres": "drama"}, {"actors": None},
])
def test_movie_dto_rejects_invalid_catalog_metadata(values):
    with pytest.raises(ValidationError):
        make_movie_dto(**values)


def test_movie_dto_requires_catalog_fields():
    with pytest.raises(ValidationError) as error:
        MovieDTO(id=1, title="Example")
    assert {item["loc"][0] for item in error.value.errors()} == {
        "tmdb_id", "overview", "genres", "release_date", "runtime",
        "vote_average", "vote_count",
    }


def test_movie_dto_accepts_zero_rating_and_independent_empty_lists():
    first = make_movie_dto(vote_average=0, vote_count=0)
    second = make_movie_dto()
    first.actors.append("actor")
    assert second.actors == []
    assert first.vote_average == 0.0
