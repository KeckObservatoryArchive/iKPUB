"""Load publications from MongoDB with full text merged from the filesystem.

Used as a library by the training and prediction pipelines:

    from data.load_pubs import load_pubs
    df = load_pubs(collection, year_start=2024, year_end=2025)
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
FULL_TEXT_DIR = PROJECT_ROOT / "data" / "pubs" / "full_text"


def load_full_text(full_text_dir: Path) -> pd.DataFrame:
    """Load full-text files from year subdirectories under full_text_dir."""
    records = []
    for subdir in sorted(full_text_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.isdigit():
            continue
        for f in sorted(subdir.glob("*.txt")):
            records.append({"bibcode": f.stem, "full": f.read_text(encoding="utf-8")})
    return pd.DataFrame(records)


def merge_full_text(df: pd.DataFrame, full_text_dir: Path) -> pd.DataFrame:
    """Left-join full text onto publications by bibcode."""
    df = df.drop(columns=["full"], errors="ignore")
    full = load_full_text(full_text_dir)
    return df.merge(full, on="bibcode", how="left")


def load_pubs(collection, year_start: int | None = None,
              year_end: int | None = None) -> pd.DataFrame:
    """Load publications from MongoDB, merge full text from filesystem.

    Returns a DataFrame ready for classifiers (has 'full' column from disk).
    """
    query = {}
    if year_start is not None:
        query["year"] = {"$gte": year_start, "$lte": year_end or year_start}

    df = pd.DataFrame(list(collection.find(query)))
    if df.empty:
        return df

    df = merge_full_text(df, FULL_TEXT_DIR)
    return df
