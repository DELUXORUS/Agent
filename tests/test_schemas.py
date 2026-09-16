from app.schemas import MovieDTO
from datetime import date

from app.db.models import Movie


def test_movie_dto_validation(sample_movie_dto):
    assert sample_movie_dto.id == 1
    assert sample_movie_dto.title == "Interstellar"
    assert sample_movie_dto.vote_average == 8.4


def test_movie_dto_converts_orm_date():
    movie = Movie(id=1, title="Interstellar", release_date=date(2014, 11, 7))
    dto = MovieDTO.model_validate(movie)
    assert dto.release_date == "2014-11-07"

