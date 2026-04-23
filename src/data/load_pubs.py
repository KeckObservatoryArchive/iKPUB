"""Load publications from MongoDB with full text merged from the filesystem.

Used as a library by the training and prediction pipelines:

    from data.load_pubs import load_pubs
    df = load_pubs(collection, year_start=2024, year_end=2025)
"""

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parents[2]
FULL_TEXT_DIR = PROJECT_ROOT / "data" / "pubs" / "full_text"


def build_subset_query(config_path: Path) -> dict:
    """Build a Mongo query from an article-subset YAML config.

    Config format:
        ranges:
          - years: [start, end]   # end may be null for open-ended
            exclude:              # optional field-value exclusions
              <field>: <value>
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    clauses = []
    for rng in config.get("ranges", []):
        start, end = rng["years"]
        year_filter = {"$gte": start}
        if end is not None:
            year_filter["$lte"] = end
        clause = {"year": year_filter}
        for field, value in (rng.get("exclude") or {}).items():
            clause[field] = {"$ne": value}
        clauses.append(clause)

    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$or": clauses}


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
              year_end: int | None = None,
              query: dict | None = None) -> pd.DataFrame:
    """Load publications from MongoDB, merge full text from filesystem.

    If ``query`` is given, it is used directly and year_start/year_end are ignored.
    Returns a DataFrame ready for classifiers (has 'full' column from disk).
    """
    if query is None:
        query = {}
        if year_start is not None:
            query["year"] = {"$gte": year_start, "$lte": year_end or year_start}

    df = pd.DataFrame(list(collection.find(query)))
    if df.empty:
        return df

    df = merge_full_text(df, FULL_TEXT_DIR)
    return df
