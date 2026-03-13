# Standard Library
import sqlite3
from pathlib import Path

# 3rd Party
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]

def load_publications(db_path: str, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load publications from a SQLite database into a DataFrame."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(query, con)

def load_full_text(full_text_dir: Path) -> pd.DataFrame:
    """
    Load full-text files from all {year}_{count} subdirectories under full_text_dir.

    Each subdirectory is expected to follow the naming pattern '{year}_{count}'.
    Each .txt file inside is named by bibcode and contains the full text.
    Returns a DataFrame with columns: bibcode, full.
    """
    records = []
    for subdir in sorted(full_text_dir.iterdir()):
        if not subdir.is_dir():
            continue
        parts = subdir.name.split("_")
        if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        year, count = parts
        txt_files = sorted(subdir.glob("*.txt"))
        if len(txt_files) != int(count):
            print(f"Warning: {subdir.name} expects {count} files but found {len(txt_files)}")
        for f in txt_files:
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
    full = load_full_text(full_text_dir)
    return df.merge(full, on="bibcode", how="left")

def load_pubs(db_path: str, manual: pd.DataFrame, full_text_dir: Path = None, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load, label, and merge full text for publications."""
    df = load_publications(db_path, query)
    df = merge_manual(df, manual)
    if full_text_dir is not None:
        df = merge_full_text(df, full_text_dir)
    return df


if __name__ == "__main__":
    db_path = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
    manual_db_path = PROJECT_ROOT / "data" / "pubs" / "manual_kpub.db"
    full_text_dir = PROJECT_ROOT / "data" / "pubs" / "full_text"

    # Load
    df = load_publications(str(db_path))

    # Keck manual labels from manual_kpub.db
    with sqlite3.connect(str(manual_db_path)) as con:
        manual = pd.read_sql("SELECT bibcode FROM pubs WHERE mission = 'keck'", con)
    df["keck_manual"] = df["bibcode"].isin(manual["bibcode"])

    # Full text
    df = merge_full_text(df, full_text_dir)

    # Write back
    with sqlite3.connect(str(db_path)) as con:
        df.to_sql("publications", con, if_exists="replace", index=False)

    print(f"Preprocessed {len(df)} publications → {db_path}")
