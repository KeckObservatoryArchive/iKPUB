"""Load publications and full text into DataFrames for classification.

SQLite:
    python src/data/prepare.py
    python src/data/prepare.py --table koa

MongoDB (no standalone step — used as a library by predict_labels.py):
    from data.prepare import load_publications_mongo
    df = load_publications_mongo(collection, year_start=2024, year_end=2025)
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]
FULL_TEXT_DIR = PROJECT_ROOT / "data" / "pubs" / "full_text"


# --- SQLite ---

def load_publications(db_path: str, query: str = "SELECT * FROM keck",
                      params: list | None = None) -> pd.DataFrame:
    """Load publications from a SQLite database into a DataFrame."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(query, con, params=params)

def load_manual_pubs(manual_db_path: str, kpub_db_path: str, table: str = "pubs",
                     year_start: int = 2000, year_end: int = 2024) -> pd.DataFrame:
    """Load article data from kpub.db with ground-truth labels from manual_kpub.db.

    Labels come from the manual DB's mission column; article content (abstract,
    title, aff, full, etc.) comes from kpub.db's keck table.
    """
    with sqlite3.connect(manual_db_path) as con:
        labels = pd.read_sql(
            f"SELECT bibcode, mission FROM {table} WHERE year >= '{year_start}' AND year <= '{year_end}'",
            con,
        )
    labels["keck_manual"] = labels["mission"] == "keck"
    labels = labels.drop(columns=["mission"])

    pubs = load_publications(kpub_db_path, "SELECT * FROM keck")
    pubs = pubs.drop(columns=["keck_manual"], errors="ignore")

    return pubs.merge(labels, on="bibcode", how="inner")


def load_full_text(full_text_dir: Path) -> pd.DataFrame:
    """Load full-text files from year subdirectories under full_text_dir."""
    records = []
    for subdir in sorted(full_text_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.isdigit():
            continue
        for f in sorted(subdir.glob("*.txt")):
            records.append({"bibcode": f.stem, "full": f.read_text(encoding="utf-8")})
    return pd.DataFrame(records)


def merge_manual(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean 'keck_manual' column — True if the bibcode appears in manual["BIBCODE"]."""
    df = df.copy()
    df["keck_manual"] = df["bibcode"].isin(manual["BIBCODE"])
    return df

def merge_full_text(df: pd.DataFrame, full_text_dir: Path) -> pd.DataFrame:
    """Left-join full text onto publications by bibcode."""
    df = df.drop(columns=["full"], errors="ignore")
    full = load_full_text(full_text_dir)
    return df.merge(full, on="bibcode", how="left")

def load_pubs(db_path: str, manual: pd.DataFrame, full_text_dir: Path = None, query: str = "SELECT * FROM keck") -> pd.DataFrame:
    """Load, label, and merge full text for publications."""
    df = load_publications(db_path, query)
    df = merge_manual(df, manual)
    if full_text_dir is not None:
        df = merge_full_text(df, full_text_dir)
    return df


MANUAL_LABEL_QUERIES = {
    "keck": "SELECT bibcode FROM pubs WHERE mission = 'keck'",
    "koa": "SELECT bibcode FROM koa",
}


# --- MongoDB ---

def load_publications_mongo(collection, year_start: int | None = None,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare publications with labels and full text")
    parser.add_argument("--table", choices=MANUAL_LABEL_QUERIES.keys(), default="keck",
                        help="table to prepare (default: keck)")
    args = parser.parse_args()

    db_path = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
    manual_db_path = PROJECT_ROOT / "data" / "pubs" / "manual_kpub.db"
    full_text_dir = PROJECT_ROOT / "data" / "pubs" / "full_text"

    # Load
    df = load_publications(str(db_path), f"SELECT * FROM {args.table}")

    # Keck manual labels from manual_kpub.db
    with sqlite3.connect(str(manual_db_path)) as con:
        manual = pd.read_sql(MANUAL_LABEL_QUERIES[args.table], con)
    df["keck_manual"] = df["bibcode"].isin(manual["bibcode"])

    # Full text
    df = merge_full_text(df, full_text_dir)

    # Write back
    with sqlite3.connect(str(db_path)) as con:
        df.to_sql(args.table, con, if_exists="replace", index=False)

    print(f"Preprocessed {len(df)} {args.table} → {db_path}")
