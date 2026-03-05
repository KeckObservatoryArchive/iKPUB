import json
import sqlite3
import os
import requests
import dotenv
from pathlib import Path

# Query 0 - Get all papers that say "Keck"
Q = "*:*"
FQ = ["year:2025", "collection:astronomy", "full:Keck", "property:refereed"]

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
    "data_facet":          ("TEXT", True),
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
    "reader":              ("TEXT", True),
    "doctype":             ("TEXT", False),
    "doctype_facet_hier":  ("TEXT", True),
    "title":               ("TEXT", True),
    "pub_raw":             ("TEXT", False),
    "property":            ("TEXT", True),
    "author":              ("TEXT", True),
    "email":               ("TEXT", True),
    "orcid":               ("TEXT", True),
    "keyword":             ("TEXT", True),
    "author_norm":         ("TEXT", True),
    "cite_read_boost":     ("REAL", False),
    "database":            ("TEXT", True),
    "classic_factor":      ("REAL", False),
    "ack":                 ("TEXT", False),
    "page":                ("TEXT", True),
    "first_author":        ("TEXT", False),
    "read_count":          ("INTEGER", False),
    "indexstamp":          ("TEXT", False),
    "issue":               ("TEXT", False),
    "keyword_facet":       ("TEXT", True),
    "aff":                 ("TEXT", True),
    "facility":            ("TEXT", True),
    "simbid":              ("TEXT", True),
    "full":                ("TEXT", False),
}

# Derived from FIELDS
FL = list(FIELDS.keys())
ARRAY_FIELDS = {name for name, (_, is_array) in FIELDS.items() if is_array}
COLUMNS = list(FIELDS.keys())

TABLE = "test_publications"

ROWS = 2000
PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"

# --- Build and run query ---
base_url = "https://api.adsabs.harvard.edu/v1/search/query"
params = {
    "q": Q,
    "fq": FQ,
    "fl": ",".join(FL),
    "rows": ROWS,
}
headers = {"Authorization": f"Bearer {dotenv.get_key('.env', 'ADS_TOKEN')}"}
response = requests.get(base_url, params=params, headers=headers)
response.raise_for_status()
docs = response.json()["response"]["docs"]

# --- Write to SQLite ---
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
print(f"Wrote {len(docs)} records to {DB_PATH}")
