"""Ingest pipeline: query ADS, fetch full text, and prepare labeled data.

Runs all three data pipeline steps in sequence for a given query profile
and year range.

Usage:
    python src/data/ingest.py --query keck --start-year 2000 --end-year 2025
    python src/data/ingest.py --query koa --start-year 2008 --end-year 2025
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import dotenv
import pandas as pd

from data.query_ads import QUERY_PROFILES, query_ads_year, write_to_db
from data.fetch_full_text import run as fetch_full_text
from data.prepare import load_publications, merge_full_text, MANUAL_LABEL_QUERIES

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
MANUAL_DB_PATH = PROJECT_ROOT / "data" / "pubs" / "manual_kpub.db"
FULL_TEXT_DIR = PROJECT_ROOT / "data" / "pubs" / "full_text"

# Map query profile to DB table name
QUERY_TABLE = {
    "keck": "keck",
    "koa": "koa",
}


def main():
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Ingest pipeline: query ADS, fetch full text, prepare data")
    parser.add_argument("--query", choices=QUERY_PROFILES.keys(), default="keck",
                        help="query profile (default: keck)")
    parser.add_argument("--start-year", type=int, default=current_year,
                        help="first year to query (default: current year)")
    parser.add_argument("--end-year", type=int, default=current_year,
                        help="last year to query (default: current year)")
    parser.add_argument("--skip-query", action="store_true",
                        help="skip ADS query (use existing DB data)")
    parser.add_argument("--skip-fulltext", action="store_true",
                        help="skip full text fetch (use existing text files)")
    args = parser.parse_args()

    table = QUERY_TABLE[args.query]
    profile = QUERY_PROFILES[args.query]

    # --- Step 1: Query ADS ---
    if not args.skip_query:
        print(f"\n=== Step 1: Query ADS ({args.query}) ===")
        headers = {"Authorization": f"Bearer {dotenv.get_key('.env', 'ADS_TOKEN')}"}
        all_docs = []
        for year in range(args.start_year, args.end_year + 1):
            all_docs.extend(query_ads_year(year, profile, headers))
        write_to_db(all_docs, table=table)
        print(f"Wrote {len(all_docs)} records to {DB_PATH}")
    else:
        print("\n=== Step 1: Query ADS (skipped) ===")

    # --- Step 2: Fetch full text ---
    if not args.skip_fulltext:
        print(f"\n=== Step 2: Fetch full text ===")
        for year in range(args.start_year, args.end_year + 1):
            fetch_full_text(year, table=table)
    else:
        print("\n=== Step 2: Fetch full text (skipped) ===")

    # --- Step 3: Prepare data ---
    print(f"\n=== Step 3: Prepare data ===")
    df = load_publications(str(DB_PATH), f"SELECT * FROM {table}")

    if table in MANUAL_LABEL_QUERIES and MANUAL_DB_PATH.exists():
        with sqlite3.connect(str(MANUAL_DB_PATH)) as con:
            manual = pd.read_sql(MANUAL_LABEL_QUERIES[table], con)
        df["keck_manual"] = df["bibcode"].isin(manual["bibcode"])
        print(f"  Labeled {df['keck_manual'].sum()} keck publications")
    else:
        print(f"  No manual labels for table '{table}', skipping labeling")

    df = merge_full_text(df, FULL_TEXT_DIR)

    with sqlite3.connect(str(DB_PATH)) as con:
        df.to_sql(table, con, if_exists="replace", index=False)

    print(f"Preprocessed {len(df)} {table} records → {DB_PATH}")


if __name__ == "__main__":
    main()
