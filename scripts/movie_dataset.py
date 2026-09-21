from ast import literal_eval
from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.core.text_normalization import normalize_search_text


@dataclass(frozen=True, slots=True)
class MovieSeedRecord:
    tmdb_id: int
    imdb_id: str | None

    title: str
    normalized_title: str
    original_title: str | None
    normalized_original_title: str | None
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


def normalize_optional_search_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    normalized = normalize_search_text(value)

    return normalized or None


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

        name = normalize_search_text(name)

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

        name = normalize_search_text(name)

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


def clean_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    text = str(value).strip()

    return text or None


def parse_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return None

    normalized = str(value).strip().lower()

    if normalized == "true":
        return True

    if normalized == "false":
        return False

    return None


def prepare_and_filter_movies(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_columns = {
        "id",
        "imdb_id",
        "title",
        "original_title",
        "original_language",
        "overview",
        "tagline",
        "genres",
        "release_date",
        "runtime",
        "vote_average",
        "vote_count",
        "adult",
        "video",
        "status",
        "actors",
        "directors",
        "keywords",
    }

    missing_columns = required_columns.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Merged movie dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = dataframe.copy()

    for column in (
        "imdb_id",
        "title",
        "original_title",
        "original_language",
        "overview",
        "tagline",
        "status",
    ):
        prepared[column] = prepared[column].map(clean_optional_text)

    prepared["normalized_title"] = prepared["title"].map(
        normalize_optional_search_text
    )
    prepared["normalized_original_title"] = prepared[
        "original_title"
    ].map(normalize_optional_search_text)

    prepared["genres"] = prepared["genres"].map(extract_names)
    prepared["adult"] = prepared["adult"].map(parse_optional_bool)
    prepared["video"] = prepared["video"].map(parse_optional_bool)

    prepared["release_date"] = pd.to_datetime(
        prepared["release_date"],
        errors="coerce",
    ).dt.date
    prepared["runtime"] = pd.to_numeric(
        prepared["runtime"],
        errors="coerce",
    )
    prepared["vote_average"] = pd.to_numeric(
        prepared["vote_average"],
        errors="coerce",
    )
    prepared["vote_count"] = pd.to_numeric(
        prepared["vote_count"],
        errors="coerce",
    )

    accepted_rows = pd.Series(
        True,
        index=prepared.index,
        dtype=bool,
    )
    report: dict[str, int] = {
        "input_rows": len(prepared),
    }

    def apply_rule(rule_name: str, condition: pd.Series) -> None:
        nonlocal accepted_rows

        condition = condition.fillna(False).astype(bool)
        rejected_rows = accepted_rows & ~condition
        report[rule_name] = int(rejected_rows.sum())
        accepted_rows &= condition

    apply_rule("adult", prepared["adult"].eq(False))
    apply_rule("video", prepared["video"].eq(False))
    apply_rule("not_released", prepared["status"].eq("Released"))
    apply_rule("missing_title", prepared["title"].notna())
    apply_rule(
        "short_or_missing_overview",
        prepared["overview"].map(
            lambda value: (
                isinstance(value, str) and len(value) >= 80
            )
        ),
    )
    apply_rule(
        "invalid_release_date",
        prepared["release_date"].notna(),
    )
    apply_rule(
        "invalid_runtime",
        prepared["runtime"].between(40, 300)
        & prepared["runtime"].mod(1).eq(0),
    )
    apply_rule(
        "invalid_vote_average",
        prepared["vote_average"].between(0, 10),
    )
    apply_rule(
        "insufficient_vote_count",
        prepared["vote_count"].ge(50)
        & prepared["vote_count"].mod(1).eq(0),
    )
    apply_rule(
        "missing_genres",
        prepared["genres"].map(bool),
    )
    apply_rule(
        "tv_movie",
        prepared["genres"].map(
            lambda genres: "tv movie" not in genres
        ),
    )

    prepared = prepared.loc[accepted_rows].copy()
    prepared["runtime"] = prepared["runtime"].astype("int64")
    prepared["vote_average"] = prepared["vote_average"].astype(float)
    prepared["vote_count"] = prepared["vote_count"].astype("int64")
    prepared = prepared.rename(columns={"id": "tmdb_id"})

    prepared = prepared[
        [
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
        ]
    ].reset_index(drop=True)

    report["output_rows"] = len(prepared)
    report["removed_rows"] = report["input_rows"] - report["output_rows"]

    return prepared, report


def row_to_movie_seed_record(row: pd.Series) -> MovieSeedRecord:
    return MovieSeedRecord(
        tmdb_id=int(row["tmdb_id"]),
        imdb_id=clean_optional_text(row["imdb_id"]),
        title=row["title"],
        normalized_title=row["normalized_title"],
        original_title=clean_optional_text(row["original_title"]),
        normalized_original_title=clean_optional_text(
            row["normalized_original_title"]
        ),
        original_language=clean_optional_text(row["original_language"]),
        overview=row["overview"],
        tagline=clean_optional_text(row["tagline"]),
        genres=tuple(row["genres"]),
        actors=tuple(row["actors"]),
        directors=tuple(row["directors"]),
        keywords=tuple(row["keywords"]),
        release_date=row["release_date"],
        runtime=int(row["runtime"]),
        vote_average=float(row["vote_average"]),
        vote_count=int(row["vote_count"]),
    )


def build_movie_seed_records(
        dataframe: pd.DataFrame
) -> list[MovieSeedRecord]:
    return [
        row_to_movie_seed_record(row)
        for _, row in dataframe.iterrows()
    ]


def build_movie_embedding_text(
    movie: MovieSeedRecord,
) -> str:
    """Build semantic text; numeric constraints remain in SQL filters."""
    parts = [f"Title: {movie.title}", f"Overview: {movie.overview}"]

    if (
        movie.original_title
        and normalize_search_text(movie.original_title)
        != normalize_search_text(movie.title)
    ):
        parts.append(f"Original title: {movie.original_title}")
    if movie.tagline:
        parts.append(f"Tagline: {movie.tagline}")
    if movie.genres:
        parts.append(f"Genres: {', '.join(movie.genres)}")
    if movie.keywords:
        parts.append(f"Keywords: {', '.join(movie.keywords[:25])}")

    return "\n".join(parts)
