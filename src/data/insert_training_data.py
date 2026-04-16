"""Ote-time insert of broad-query records from SQLite into MongoDB as training data.

Reads all rows from the `keck` table in data/pubs/kpub.db and inserts any whose
bibcode is not already present in the target MongoDB collection. Inserted docs
are marked with `from_broad_query: true` and `affiliation: "unknown"` so they
can be used as negative/unlabeled training examples.

Usage:
    python -m src.data.insert_training_data
    python -m src.data.insert_training_data --collection test_articles
    python -m src.data.insert_training_data --dry-run
"""

import argparse
import json
import sqlite3
from pathlib import Path

from data.db_mongo_conn import from_env

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"

MONGO_DB = "kpub"
DEFAULT_COLLECTION = "test_articles"
BATCH_SIZE = 1000

DROP_FIELDS = {"embedding", "full", "keck_manual"}


def normalize(val):
    """Deserialize JSON-encoded arrays/objects stored as strings in SQLite."""
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except Exception:
                pass
    return val


def build_doc(row: dict) -> dict:
    doc = {
        k: normalize(v)
        for k, v in row.items()
        if v is not None and k not in DROP_FIELDS
    }
    if "year" in doc:
        doc["year"] = int(doc["year"])
    if "date" in doc and isinstance(doc["date"], str) and len(doc["date"]) >= 7:
        doc["month"] = int(doc["date"][5:7])
    doc["_id"] = doc["bibcode"]
    doc["from_broad_query"] = True
    doc["affiliation"] = "unknown"
    doc["last_modifier"] = "ikpub"
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"Target MongoDB collection (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts without writing to MongoDB")
    args = parser.parse_args()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM keck")]
    con.close()
    print(f"Loaded {len(rows)} rows from SQLite keck table")

    connector = from_env(MONGO_DB, args.collection)
    col = connector.collection
    existing = set(
        d["bibcode"] for d in col.find({}, {"bibcode": 1}) if "bibcode" in d
    )
    print(f"Found {len(existing)} existing docs in {args.collection}")

    new_rows = [r for r in rows if r.get("bibcode") and r["bibcode"] not in existing]
    print(f"To insert: {len(new_rows)}")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    if not new_rows:
        return

    inserted = 0
    for i in range(0, len(new_rows), BATCH_SIZE):
        batch = [build_doc(r) for r in new_rows[i:i + BATCH_SIZE]]
        col.insert_many(batch, ordered=False)
        inserted += len(batch)
        print(f"  inserted {inserted}/{len(new_rows)}")

    print(f"\nInserted: {inserted}")


if __name__ == "__main__":
    main()
