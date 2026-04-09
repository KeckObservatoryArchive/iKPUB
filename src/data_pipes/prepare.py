# Standard Library
import argparse
import sqlite3
from pathlib import Path

# 3rd Party
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]

def load_publications(db_path: str, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load publications from a SQLite database into a DataFrame."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(query, con)

def load_manual_pubs(manual_db_path: str, kpub_db_path: str, table: str = "pubs",
                     year_start: int = 2000, year_end: int = 2024) -> pd.DataFrame:
    """Load article data from kpub.db with ground-truth labels from manual_kpub.db.

    Labels come from the manual DB's mission column; article content (abstract,
    title, aff, full, etc.) comes from kpub.db's publications table.
    """
    with sqlite3.connect(manual_db_path) as con:
        labels = pd.read_sql(
            f"SELECT bibcode, mission FROM {table} WHERE year >= '{year_start}' AND year <= '{year_end}'",
            con,
        )
    labels["keck_manual"] = labels["mission"] == "keck"
    labels = labels.drop(columns=["mission"])

    pubs = load_publications(kpub_db_path, "SELECT * FROM publications")
    pubs = pubs.drop(columns=["keck_manual"], errors="ignore")

    return pubs.merge(labels, on="bibcode", how="inner")


def load_full_text(full_text_dir: Path) -> pd.DataFrame:
    """
    Load full-text files from year subdirectories under full_text_dir.

    Each subdirectory is expected to be named by year (e.g. '2022').
    Each .txt file inside is named by bibcode and contains the full text.
    Returns a DataFrame with columns: bibcode, full.
    """
    records = []
    for subdir in sorted(full_text_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.isdigit():
            continue
        for f in sorted(subdir.glob("*.txt")):
            records.append({"bibcode": f.stem, "full": f.read_text(encoding="utf-8")})
    return pd.DataFrame(records)


def merge_manual(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'keck_manual' column — True if the bibcode appears in manual["BIBCODE"].

    Expects df to have a 'bibcode' column and manual to have a 'BIBCODE' column.
    """
    df = df.copy()
    df["keck_manual"] = df["bibcode"].isin(manual["BIBCODE"])
    return df

def merge_full_text(df: pd.DataFrame, full_text_dir: Path) -> pd.DataFrame:
    """Left-join full text onto publications by bibcode."""
    df = df.drop(columns=["full"], errors="ignore")
    full = load_full_text(full_text_dir)
    return df.merge(full, on="bibcode", how="left")

def load_pubs(db_path: str, manual: pd.DataFrame, full_text_dir: Path = None, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load, label, and merge full text for publications."""
    df = load_publications(db_path, query)
    df = merge_manual(df, manual)
    if full_text_dir is not None:
        df = merge_full_text(df, full_text_dir)
    return df


MANUAL_LABEL_QUERIES = {
    "publications": "SELECT bibcode FROM pubs WHERE mission = 'keck'",
    "koa": "SELECT bibcode FROM koa",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare publications with labels and full text")
    parser.add_argument("--table", choices=MANUAL_LABEL_QUERIES.keys(), default="publications",
                        help="table to prepare (default: publications)")
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
