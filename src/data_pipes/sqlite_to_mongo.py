"""Transfer prediction labels from SQLite to MongoDB.

Reads the predictions table from kpub.db and upserts ilabel + confidence
into MongoDB. Bibcodes already in Mongo get updated; new bibcodes get
inserted as full documents.

Usage:
    python -m data_pipes.sqlite_to_mongo
    python -m data_pipes.sqlite_to_mongo 2024
    python -m data_pipes.sqlite_to_mongo 2020-2024
"""

import argparse
import json
import sqlite3
from pathlib import Path

from pymongo import MongoClient, UpdateOne

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "mongo_dump"
MONGO_COLLECTION = "articles"


def parse_year_arg(year_str: str) -> tuple[int, int]:
    if "-" in year_str:
        start, end = year_str.split("-", 1)
        return int(start), int(end)
    year = int(year_str)
    return year, year


def normalize(val):
    """Deserialize JSON-encoded strings (arrays/objects) from SQLite."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.strip()
        if val.startswith("[") or val.startswith("{"):
            try:
                return json.loads(val)
            except Exception:
                pass
    return val


def main():
    parser = argparse.ArgumentParser(description="Transfer predictions to MongoDB.")
    parser.add_argument("year", nargs="?", default=None,
                        help="Optional: single year (2024) or range (2020-2024)")
    args = parser.parse_args()

    # --- Read predictions from SQLite ---
    query = "SELECT * FROM predictions"
    if args.year:
        y_start, y_end = parse_year_arg(args.year)
        query += f" WHERE year >= {y_start} AND year <= {y_end}"

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(query)]
    con.close()

    if not rows:
        print("No predictions found. Exiting.")
        return

    print(f"Loaded {len(rows)} predictions from SQLite")

    # --- Connect to Mongo and prefetch existing bibcodes ---
    client = MongoClient(MONGO_URI)
    col = client[MONGO_DB][MONGO_COLLECTION]

    existing = set(
        d["bibcode"] for d in col.find({}, {"bibcode": 1}) if "bibcode" in d
    )
    print(f"Found {len(existing)} existing docs in MongoDB")

    # --- Split into updates vs inserts ---
    updates = []
    inserts = []

    for row in rows:
        bibcode = row.get("bibcode")
        if not bibcode:
            continue

        if bibcode in existing:
            updates.append(UpdateOne(
                {"bibcode": bibcode},
                {"$set": {"ilabel": row["ilabel"], "confidence": row["confidence"]}},
            ))
        else:
            doc = {k: normalize(v) for k, v in row.items() if v is not None}
            if "year" in doc:
                doc["year"] = int(doc["year"])
            inserts.append(doc)

    # --- Write to Mongo ---
    updated_count = 0
    inserted_count = 0

    if updates:
        result = col.bulk_write(updates)
        updated_count = result.modified_count

    if inserts:
        col.insert_many(inserts)
        inserted_count = len(inserts)

    print(f"\nUpdated: {updated_count}")
    print(f"Inserted: {inserted_count}")
    print(f"Total processed: {len(rows)}")


if __name__ == "__main__":
    main()
