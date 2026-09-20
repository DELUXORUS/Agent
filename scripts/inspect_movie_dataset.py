from pathlib import Path

import pandas as pd

from scripts.movie_dataset import (
    build_movie_embedding_text,
    deduplicate_movies,
    merge_movie_datasets,
    normalize_id_column,
    prepare_and_filter_movies,
    prepare_credits,
    prepare_keywords,
    build_movie_seed_records
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "the_movies_dataset"


def load_raw_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    movies = pd.read_csv(
        DATASET_DIR / "movies_metadata.csv",
        dtype={"id": "string"},
        low_memory=False,
    )
    credits = pd.read_csv(
        DATASET_DIR / "credits.csv",
        dtype={"id": "string"},
    )
    keywords = pd.read_csv(
        DATASET_DIR / "keywords.csv",
        dtype={"id": "string"},
    )

    movies, invalid_movie_ids = normalize_id_column(movies)
    credits, invalid_credit_ids = normalize_id_column(credits)
    keywords, invalid_keyword_ids = normalize_id_column(keywords)

    credits, merged_credit_rows = prepare_credits(credits)
    keywords, merged_keyword_rows = prepare_keywords(keywords)

    print(
        f"Merged duplicate credit rows: "
        f"{merged_credit_rows}"
    )
    print(
        f"Merged duplicate keyword rows: "
        f"{merged_keyword_rows}"
    )

    movies, removed_duplicate_movies = deduplicate_movies(movies)

    print(f"Removed invalid movie IDs: {invalid_movie_ids}")
    print(f"Removed invalid credit IDs: {invalid_credit_ids}")
    print(f"Removed invalid keyword IDs: {invalid_keyword_ids}")
    print(f"Removed duplicate movie rows: {removed_duplicate_movies}")

    return movies, credits, keywords


def print_dataset_info(name: str, dataframe: pd.DataFrame) -> None:
    print(f"\n{name}")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {dataframe.columns.tolist()}")
    print(dataframe.head(2))


def main() -> None:
    movies, credits, keywords = load_raw_datasets()

    print_dataset_info("Movies", movies)
    print_dataset_info("Credits", credits)
    print_dataset_info("Keywords", keywords)

    for name, dataframe in (
            ("movies", movies),
            ("credits", credits),
            ("keywords", keywords),
    ):
        numeric_ids = pd.to_numeric(
            dataframe["id"],
            errors="coerce",
        )

        invalid_ids_count = numeric_ids.isna().sum()
        duplicate_ids_count = numeric_ids.dropna().duplicated().sum()

        print(
            f"{name}: "
            f"invalid IDs = {invalid_ids_count}, "
            f"duplicate IDs = {duplicate_ids_count}"
        )

    merged_movies = merge_movie_datasets(
        movies,
        credits,
        keywords,
    )

    print_dataset_info("Merged movies", merged_movies)
    print("Merged movie rows:", len(merged_movies))
    print(
        "Movies without actors:",
        merged_movies["actors"].map(len).eq(0).sum(),
    )
    print(
        "Movies without directors:",
        merged_movies["directors"].map(len).eq(0).sum(),
    )
    print(
        "Movies without keywords:",
        merged_movies["keywords"].map(len).eq(0).sum(),
    )

    prepared_movies, filter_report = prepare_and_filter_movies(
        merged_movies
    )

    print_dataset_info("Prepared movies", prepared_movies)
    print("Filter report:")

    for rule_name, rows_count in filter_report.items():
        print(f"  {rule_name}: {rows_count}")

    the_game = prepared_movies.loc[
        prepared_movies["tmdb_id"].eq(2649)
    ]
    print("The Game is present:", not the_game.empty)

    records = build_movie_seed_records(prepared_movies)

    print("Количество записей:", len(records))

    if records:
        print("Первый фильм:", records[0])
        print("Текст для embedding:\n", build_movie_embedding_text(records[0]))


if __name__ == "__main__":
    main()

