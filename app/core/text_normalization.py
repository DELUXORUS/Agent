import unicodedata


def normalize_search_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Search text must be a string")

    unicode_normalized = unicodedata.normalize("NFKC", value)
    whitespace_normalized = " ".join(unicode_normalized.split())

    return whitespace_normalized.casefold()
