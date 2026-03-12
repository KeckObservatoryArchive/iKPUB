"""
Query ads for publications. Change the query using the constants at the top of this file.
"""

import argparse
import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path

import requests
import dotenv

# Query 0 - Get all papers that say "Keck"
Q = "*:*"
BASE_FQ = ["collection:astronomy", "full:Keck", "property:refereed"]

# Single source of truth: field_name → (sqlite_type, is_array)
# Add/remove fields here — everything else adapts automatically.
FIELDS = {
    "bibcode":             ("TEXT PRIMARY KEY", False),
    "date":                ("TEXT", False),
    "pub":                 ("TEXT", False),
    "id":                  ("TEXT", False),
    "volume":              ("TEXT", False),
    "links_data":          ("TEXT", True),
    "citation":            ("TEXT", True),
    "doi":                 ("TEXT", True),
    "eid":                 ("TEXT", False),
    "keyword_schema":      ("TEXT", True),
    "citation_count":      ("INTEGER", False),
    "data":                ("TEXT", True),
    "year":                ("TEXT", False),
    "identifier":          ("TEXT", True),
    "keyword_norm":        ("TEXT", True),
    "reference":           ("TEXT", True),
    "abstract":            ("TEXT", False),
    "recid":               ("INTEGER", False),
    "alternate_bibcode":   ("TEXT", True),
    "arxiv_class":         ("TEXT", True),
    "first_author_norm":   ("TEXT", False),
    "pubdate":             ("TEXT", False),
    "doctype":             ("TEXT", False),
    "doctype_facet_hier":  ("TEXT", True),
    "title":               ("TEXT", True),
    "pub_raw":             ("TEXT", False),
    "property":            ("TEXT", True),
    "author":              ("TEXT", True),
    "email":               ("TEXT", True),
    "keyword":             ("TEXT", True),
    "author_norm":         ("TEXT", True),
    "cite_read_boost":     ("REAL", False),
    "database":            ("TEXT", True),
    "classic_factor":      ("REAL", False),
    "page":                ("TEXT", True),
    "first_author":        ("TEXT", False),
    "read_count":          ("INTEGER", False),
    "indexstamp":          ("TEXT", False),
    "issue":               ("TEXT", False),
    "keyword_facet":       ("TEXT", True),
    "aff":                 ("TEXT", True),
    "facility":            ("TEXT", True),
    "simbid":              ("TEXT", True),
}

# Derived from FIELDS
FL = list(FIELDS.keys())
ARRAY_FIELDS = {name for name, (_, is_array) in FIELDS.items() if is_array}
COLUMNS = list(FIELDS.keys())

TABLE = "test_publications"

ROWS = 2000
PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"


def query_ads_year(year: int, headers: dict) -> list[dict]:
    """Query ADS for a single year and return the docs."""
    base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    fq = [f"year:{year}"] + BASE_FQ
    params = {
        "q": Q,
        "fq": fq,
        "fl": ",".join(FL),
        "rows": ROWS,
    }
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    docs = response.json()["response"]["docs"]
    print(f"  {year}: {len(docs)} records")
    return docs


def write_to_db(docs: list[dict]) -> None:
    """Write docs to SQLite."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    col_defs = ", ".join(f"{name} {sql_type}" for name, (sql_type, _) in FIELDS.items())
    cur.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({col_defs})")

    placeholders = ", ".join("?" * len(COLUMNS))
    col_list = ", ".join(COLUMNS)

    for doc in docs:
        values = []
        for col in COLUMNS:
            val = doc.get(col)
            if val is None:
                values.append(None)
            elif col in ARRAY_FIELDS:
                values.append(json.dumps(val) if isinstance(val, list) else val)
            else:
                values.append(val)
        cur.execute(
            f"INSERT OR REPLACE INTO {TABLE} ({col_list}) VALUES ({placeholders})",
            values,
        )
    con.commit()
    con.close()


def main():
    current_year = datetime.now().year
    parser = argparse.ArgumentParser(description="Query ADS for Keck publications")
    parser.add_argument("--start-year", type=int, default=current_year, help="first year to query (default: current year)")
    parser.add_argument("--end-year", type=int, default=current_year, help="last year to query (default: current year)")
    args = parser.parse_args()

    headers = {"Authorization": f"Bearer {dotenv.get_key('.env', 'ADS_TOKEN')}"}

    all_docs = []
    for year in range(args.start_year, args.end_year + 1):
        all_docs.extend(query_ads_year(year, headers))

    write_to_db(all_docs)
    print(f"Wrote {len(all_docs)} records to {DB_PATH}")


if __name__ == "__main__":
    main()
