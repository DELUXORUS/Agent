from app.schemas import MovieDTO
from app.agent.state import MovieFilter, IntentClassification


def test_movie_dto_validation(sample_movie_dto):
    """Проверка корректной сборки DTO."""
    assert sample_movie_dto.id == 1
    assert sample_movie_dto.title == "Interstellar"
    assert sample_movie_dto.vote_average == 8.4


def test_movie_filter_defaults():
    """Проверка значений по умолчанию в фильтре."""
    filter_obj = MovieFilter(query_text="space adventure")
    assert filter_obj.query_text == "space adventure"
    assert filter_obj.is_semantic_search_needed is True
    assert filter_obj.genre is None
    assert filter_obj.min_vote_average is None


def test_intent_classification_schema():
    """Проверка допустимых значений интентов."""
    intent = IntentClassification(intent="guess_movie")
    assert intent.intent == "guess_movie"