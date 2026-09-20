from ast import literal_eval
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True, slots=True)
class MovieSeedRecord:
    tmdb_id: int
    imdb_id: str | None

    title: str
    original_title: str | None
    original_language: str | None

    overview: str
    tagline: str | None

    genres: tuple[str, ...]
    actors: tuple[str, ...]
    directors: tuple[str, ...]
    keywords: tuple[str, ...]

    release_date: date
    runtime: int

    vote_average: float
    vote_count: int


def parse_list_of_dicts(value: object) -> list[dict[str, object]]:
    if value is None:
        return []

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    try:
        parsed_value = literal_eval(value)
    except (SyntaxError, ValueError) as error:
        raise ValueError("Cannot parse dataset value as a list") from error

    if not isinstance(parsed_value, list):
        raise ValueError("Dataset value must contain a list")

    if not all(isinstance(item, dict) for item in parsed_value):
        raise ValueError("Dataset list must contain dictionaries")

    return parsed_value


def extract_names(value: object) -> tuple[str, ...]:
    items = parse_list_of_dicts(value)

    names: list[str] = []
    seen_names: set[str] = set()

    for item in items:
        name = item.get("name")

        if not isinstance(name, str):
            continue

        name = name.strip()

        if not name or name in seen_names:
            continue

        names.append(name)
        seen_names.add(name)

    return tuple(names)


def extract_directors(value: object) -> tuple[str, ...]:
    crew_items = parse_list_of_dicts(value)

    director_names: list[str] = []
    seen_names: set[str] = set()

    for item in crew_items:
        if item.get("job") != "Director":
            continue

        name = item.get("name")

        if not isinstance(name, str):
            continue

        name = name.strip()

        if not name or name in seen_names:
            continue

        director_names.append(name)
        seen_names.add(name)

    return tuple(director_names)


def normalize_id_column(
        dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    if "id" not in dataframe.columns:
        raise ValueError("Dataset must contain an 'id' column")

    normalized_dataframe = dataframe.copy()

    numeric_ids = pd.to_numeric(
        normalized_dataframe["id"],
        errors="coerce",
    )

    invalid_ids_count = int(numeric_ids.isna().sum())
    valid_ids_mask = numeric_ids.notna()

    normalized_dataframe = normalized_dataframe.loc[
        valid_ids_mask
    ].copy()

    normalized_dataframe["id"] = numeric_ids.loc[
        valid_ids_mask
    ].astype("int64")

    return normalized_dataframe, invalid_ids_count


def deduplicate_movies(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required_columns = {
        "id",
        "overview",
        "vote_count",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Movies dataset is missing columns: {sorted(missing_columns)}"
        )

    deduplicated = dataframe.copy()
    rows_before = len(deduplicated)

    deduplicated["_numeric_vote_count"] = pd.to_numeric(
        deduplicated["vote_count"],
        errors="coerce",
    ).fillna(-1)

    deduplicated["_overview_length"] = (
        deduplicated["overview"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    deduplicated = (
        deduplicated
        .sort_values(
            by=[
                "id",
                "_numeric_vote_count",
                "_overview_length",
            ],
            ascending=[True, False, False],
            kind="stable",
        )
        .drop_duplicates(
            subset="id",
            keep="first",
        )
        .drop(
            columns=[
                "_numeric_vote_count",
                "_overview_length",
            ]
        )
        .reset_index(drop=True)
    )

    removed_rows_count = rows_before - len(deduplicated)

    return deduplicated, removed_rows_count


def merge_unique_names(
    values: pd.Series,
) -> tuple[str, ...]:
    merged_names: list[str] = []
    seen_names: set[str] = set()

    for names in values:
        for name in names:
            if name in seen_names:
                continue

            merged_names.append(name)
            seen_names.add(name)

    return tuple(merged_names)


def prepare_credits(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required_columns = {
        "id",
        "cast",
        "crew",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Credits dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = dataframe[["id"]].copy()
    rows_before = len(prepared)

    prepared["actors"] = dataframe["cast"].map(extract_names)
    prepared["directors"] = dataframe["crew"].map(
        extract_directors
    )

    prepared = (
        prepared
        .groupby(
            "id",
            as_index=False,
            sort=False,
        )
        .agg(
            actors=("actors", merge_unique_names),
            directors=("directors", merge_unique_names),
        )
    )

    merged_rows_count = rows_before - len(prepared)

    return prepared, merged_rows_count


def prepare_keywords(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required_columns = {
        "id",
        "keywords",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Keywords dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = dataframe[["id"]].copy()
    rows_before = len(prepared)

    prepared["keywords"] = dataframe["keywords"].map(
        extract_names
    )

    prepared = (
        prepared
        .groupby(
            "id",
            as_index=False,
            sort=False,
        )
        .agg(
            keywords=("keywords", merge_unique_names),
        )
    )

    merged_rows_count = rows_before - len(prepared)

    return prepared, merged_rows_count


def merge_movie_datasets(
    movies: pd.DataFrame,
    credits: pd.DataFrame,
    keywords: pd.DataFrame,
) -> pd.DataFrame:
    datasets = {
        "movies": movies,
        "credits": credits,
        "keywords": keywords,
    }

    for dataset_name, dataframe in datasets.items():
        if "id" not in dataframe.columns:
            raise ValueError(
                f"{dataset_name} dataset must contain an 'id' column"
            )

        if dataframe["id"].duplicated().any():
            raise ValueError(
                f"{dataset_name} dataset contains duplicate IDs"
            )

    merged = movies.merge(
        credits,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    merged = merged.merge(
        keywords,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    )

    if len(merged) != len(movies):
        raise RuntimeError(
            "Movie count changed while merging datasets"
        )

    for column in ("actors", "directors", "keywords"):
        merged[column] = merged[column].map(
            lambda value: value if isinstance(value, tuple) else ()
        )

    return merged
