"""Batch-predict publication labels and write them to a SQLite table.

Loads a trained LegacyTransformer model, runs inference on publications
for a given year or year range, and upserts (bibcode, ilabel, confidence)
into a ``predictions`` table.

Usage:
    python -m eval.predict_labels 2023
    python -m eval.predict_labels 2020-2024
    python -m eval.predict_labels 2024 --model-path data/models/trained/my_model
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from data_pipes.prepare import load_publications
from models.transformer_legacy import LegacyTransformerClassifier

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "data" / "models" / "trained" / "transformer_2026-03-17_173704"
DEFAULT_DB = PROJECT_ROOT / "data" / "pubs" / "kpub.db"

LOWER_THRESHOLD = 0.20
UPPER_THRESHOLD = 0.75


def parse_year_arg(year_str: str) -> tuple[int, int]:
    """Parse a year or year range (e.g. '2023' or '2020-2024')."""
    if "-" in year_str:
        start, end = year_str.split("-", 1)
        return int(start), int(end)
    year = int(year_str)
    return year, year


def prob_to_ilabel(prob: float) -> str:
    if prob >= UPPER_THRESHOLD:
        return "keck"
    if prob <= LOWER_THRESHOLD:
        return "not keck"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Predict publication labels.")
    parser.add_argument("year", help="Single year (2023) or range (2020-2024)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()

    year_start, year_end = parse_year_arg(args.year)

    print(f"Loading model from {args.model_path}")
    model = LegacyTransformerClassifier.load(args.model_path)

    query = f"SELECT * FROM publications WHERE year >= {year_start} AND year <= {year_end}"
    pubs = load_publications(args.db_path, query=query)
    print(f"Loaded {len(pubs)} publications for years {year_start}-{year_end}")

    if pubs.empty:
        print("No publications found. Exiting.")
        return

    probs = model.predict(pubs, return_proba=True)

    results = pd.DataFrame({
        "bibcode": pubs["bibcode"].values,
        "ilabel": [prob_to_ilabel(p) for p in probs],
        "confidence": probs.values,
    })

    with sqlite3.connect(args.db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                bibcode TEXT PRIMARY KEY,
                ilabel TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        """)
        con.executemany(
            "INSERT OR REPLACE INTO predictions (bibcode, ilabel, confidence) VALUES (?, ?, ?)",
            results.itertuples(index=False, name=None),
        )

    summary = results["ilabel"].value_counts()
    print(f"\nWrote {len(results)} predictions to {args.db_path}::predictions")
    print(summary.to_string())


if __name__ == "__main__":
    main()
