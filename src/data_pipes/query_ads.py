"""
Query ads for publications. Select a query profile with --query (default: keck).
"""

import argparse
import json
import sqlite3
import os
from datetime import datetime
from pathlib import Path

import requests
import dotenv

# ---------------------------------------------------------------------------
# Query profiles
# ---------------------------------------------------------------------------
QUERY_PROFILES = {
    "keck": {
        "q": "*:*",
        "base_fq": ["collection:astronomy", "full:Keck", "property:refereed"],
        "fl": None,  # None = use all FIELDS
        "sort": None,
        "hl": None,
    },
    "koa": {
        "q": (
            'ack:"Keck Observatory Archive"'
            ' OR body:("archive" NEAR5 ("Keck" OR "HIRES" OR "ESI" OR "NIRC2"'
            ' OR "DEIMOS" OR "OSIRIS" OR "NIRSPEC" OR "NIRC" OR "LRIS"'
            ' OR "MOSFIRE" OR "LWS" OR "NIRES" OR "KCWI" OR "KPF" OR "KCRM"))'
        ),
        "base_fq": ["collection:astronomy", "property:article", "property:refereed"],
        "fl": None,
        "sort": "bibcode desc",
        "hl": {
            "hl": "true",
            "hl.fl": "body,ack",
            "hl.snippets": 4,
            "hl.maxAnalyzedChars": 150000,
        },
    },
}

# ---------------------------------------------------------------------------
# Single source of truth: field_name → (sqlite_type, is_array)
# Add/remove fields here — everything else adapts automatically.
# ---------------------------------------------------------------------------
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
    "ack":                 ("TEXT", False),
}

# Derived from FIELDS
ARRAY_FIELDS = {name for name, (_, is_array) in FIELDS.items() if is_array}
COLUMNS = list(FIELDS.keys())

TABLE = "koa"

ROWS = 2000
PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"


def query_ads_year(year: int, profile: dict, headers: dict) -> list[dict]:
    """Query ADS for a single year and return the docs."""
    base_url = "https://api.adsabs.harvard.edu/v1/search/query"
    fl = profile["fl"] or list(FIELDS.keys())
    fq = [f"year:{year}"] + profile["base_fq"]
    params = {
        "q": profile["q"],
        "fq": fq,
        "fl": ",".join(fl),
        "rows": ROWS,
    }
    if profile["sort"]:
        params["sort"] = profile["sort"]
    if profile["hl"]:
        params.update(profile["hl"])
    response = requests.get(base_url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    docs = data["response"]["docs"]
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
    parser.add_argument("--query", choices=QUERY_PROFILES.keys(), default="keck",
                        help="query profile to use (default: keck)")
    parser.add_argument("--start-year", type=int, default=current_year, help="first year to query (default: current year)")
    parser.add_argument("--end-year", type=int, default=current_year, help="last year to query (default: current year)")
    args = parser.parse_args()

    profile = QUERY_PROFILES[args.query]
    headers = {"Authorization": f"Bearer {dotenv.get_key('.env', 'ADS_TOKEN')}"}

    all_docs = []
    for year in range(args.start_year, args.end_year + 1):
        all_docs.extend(query_ads_year(year, profile, headers))

    write_to_db(all_docs)
    print(f"Wrote {len(all_docs)} records to {DB_PATH}")


if __name__ == "__main__":
    main()
