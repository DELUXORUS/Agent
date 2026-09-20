from sqlalchemy import CheckConstraint, Integer, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ARRAY

from app.db.models import Movie, UserMovieHistory


def test_movie_schema_has_required_catalog_columns():
    columns = Movie.__table__.c

    assert {
        "id",
        "tmdb_id",
        "imdb_id",
        "title",
        "normalized_title",
        "original_title",
        "normalized_original_title",
        "original_language",
        "overview",
        "tagline",
        "genres",
        "actors",
        "directors",
        "keywords",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "embedding",
    } == set(columns.keys())

    for column_name in (
        "tmdb_id",
        "title",
        "normalized_title",
        "overview",
        "genres",
        "actors",
        "directors",
        "keywords",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
    ):
        assert not columns[column_name].nullable


def test_movie_metadata_uses_postgresql_arrays():
    dialect = postgresql.dialect()

    for column_name in ("genres", "actors", "directors", "keywords"):
        column_type = Movie.__table__.c[column_name].type
        assert isinstance(column_type.dialect_impl(dialect), ARRAY)


def test_movie_constraints_have_stable_names():
    constraints = Movie.__table__.constraints
    unique_constraints = {
        constraint.name
        for constraint in constraints
        if isinstance(constraint, UniqueConstraint)
    }
    check_constraints = {
        constraint.name
        for constraint in constraints
        if isinstance(constraint, CheckConstraint)
    }

    assert unique_constraints == {
        "uq_movies_tmdb_id",
        "uq_movies_imdb_id",
    }
    assert check_constraints == {
        "ck_movies_runtime_positive",
        "ck_movies_vote_average_range",
        "ck_movies_vote_count_non_negative",
    }


def test_movie_search_indexes_are_configured():
    indexes = {
        index.name: index
        for index in Movie.__table__.indexes
    }

    for index_name in (
        "idx_movies_genres_gin",
        "idx_movies_actors_gin",
        "idx_movies_directors_gin",
    ):
        assert indexes[index_name].dialect_options["postgresql"]["using"] == "gin"

    hnsw = indexes["idx_movies_embedding_hnsw"]
    assert hnsw.dialect_options["postgresql"]["using"] == "hnsw"
    assert hnsw.dialect_options["postgresql"]["ops"] == {
        "embedding": "vector_cosine_ops",
    }

    assert {
        "idx_movies_normalized_title",
        "idx_movies_normalized_original_title",
        "idx_movies_vote_average",
        "idx_movies_release_date",
        "idx_movies_runtime",
    }.issubset(indexes)


def test_movie_history_foreign_key_matches_movie_primary_key_type():
    assert isinstance(Movie.__table__.c.id.type, Integer)
    assert isinstance(UserMovieHistory.__table__.c.movie_id.type, Integer)
