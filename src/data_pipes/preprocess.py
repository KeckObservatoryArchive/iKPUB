# Standard Library
import json
import re
import sqlite3
from pathlib import Path

# 3rd Party
from sentence_transformers import SentenceTransformer
import pandas as pd

PROJECT_ROOT = Path(__file__).parents[2]

# Embedding model — swap and re-run to experiment
MODEL_NAME = PROJECT_ROOT / "data" / "models" / "all-mpnet-base-v2"
# MODEL_NAME = "malteos/scincl"          # Citation-graph trained, scientific papers
# MODEL_NAME = "allenai/specter2_base"   # SPECTER2 base (no adapter); CLS pooling mismatch

def load_publications(db_path: str, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load publications from a SQLite database into a DataFrame."""
    with sqlite3.connect(db_path) as con:
        return pd.read_sql(query, con)

def clean_authors(authors_str: str) -> str:
    """Lowercase and strip non-alphabetic punctuation from an author string."""
    if not isinstance(authors_str, str):
        return ""
    cleaned = re.sub(r"[^a-z,\s\-]", "", authors_str.lower())
    return re.sub(r"\s*,\s*", ", ", cleaned).strip()

def embed(df: pd.DataFrame, model_name: str = MODEL_NAME) -> pd.DataFrame:
    """
    Add 'authors_clean' and 'embedding' columns to df.

    Expects columns: title, abstract.
    Returns the same DataFrame with 'embedding' appended.
    """
    model = SentenceTransformer(str(model_name))

    # Concatenate title and abstract; [SEP] is the separator scincl was trained on.
    # For all-mpnet-base-v2 this is benign but can be swapped to a space if preferred.
    texts = [
        f"{row['title'] or ''} [SEP] {row['abstract'] or ''}".strip()
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["embedding"] = model.encode(texts, show_progress_bar=True).tolist()

    return df

def merge_manual(df: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """
    Add a boolean 'keck_manual' column — True if the bibcode appears in manual["BIBCODE"].

    Expects df to have a 'bibcode' column and manual to have a 'BIBCODE' column.
    """
    df = df.copy()
    df["keck_manual"] = df["bibcode"].isin(manual["BIBCODE"])
    return df

def load_pubs(db_path: str, manual: pd.DataFrame, query: str = "SELECT * FROM publications") -> pd.DataFrame:
    """Load, clean, embed, and label publications end to end."""
    df = load_publications(db_path, query)
    df["authors_clean"] = df["author"].apply(clean_authors)
    df = merge_manual(df, manual)
    df = embed(df)
    return df


if __name__ == "__main__":
    db_path = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
    manual_db_path = PROJECT_ROOT / "data" / "pubs" / "manual_kpub.db"

    # Load
    df = load_publications(str(db_path))

    # Clean authors
    df["authors_clean"] = df["author"].apply(clean_authors)

    # Keck manual labels from manual_kpub.db
    with sqlite3.connect(str(manual_db_path)) as con:
        manual = pd.read_sql("SELECT bibcode FROM pubs WHERE mission = 'keck'", con)
    df["keck_manual"] = df["bibcode"].isin(manual["bibcode"])

    # Embeddings
    df = embed(df)

    # Serialize embeddings to JSON strings for SQLite
    df["embedding"] = df["embedding"].apply(json.dumps)

    # Write back
    with sqlite3.connect(str(db_path)) as con:
        df.to_sql("publications", con, if_exists="replace", index=False)

    print(f"Preprocessed {len(df)} publications → {db_path}")
