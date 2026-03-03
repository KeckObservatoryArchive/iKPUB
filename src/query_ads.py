import json
import sqlite3
import os
import requests
import dotenv

# Query 0 - Get all papers that say "Keck"
Q = "*:*"
FQ = ["year:2023", "collection:astronomy", "full:Keck", "property:refereed"]

# # Query 1 - Acknowledgement/Abstract search
# Q = '(ack:keck OR abs:keck) NOT full:BICEP NOT full:BICEP2 NOT full:NUANCE NOT full:"keck array"'
# FQ = ["collection:astronomy", "property:refereed", "-property:nonarticle", "year:2023"]

# Q = '(ack:keck OR abs:keck) -full:BICEP -full:BICEP2 -full:NUANCE -full:"keck array"'
# FQ = ["collection:astronomy", "property:refereed", "-property:nonarticle", "year:2023"]

# # Query 2 - Full text search
# Q = '(full:"keck observatory" OR full:"keck DEIMOS" OR full:"keck HIRES" OR full:"keck LRIS") NOT full:BICEP NOT full:BICEP2'
# FQ = ["collection:astronomy", "property:refereed", "-property:nonarticle", "year:2023"]

# FL = ['date', 'bibcode', 'title', 'abstract'] # Basic fields
FL = ['date', 'pub', 'id', 'volume', 'links_data', 'citation', 'doi',
        'eid', 'keyword_schema', 'citation_count', 'data', 'data_facet',
        'year', 'identifier', 'keyword_norm', 'reference', 'abstract', 'recid',
        'alternate_bibcode', 'arxiv_class', 'bibcode', 'first_author_norm',
        'pubdate', 'reader', 'doctype', 'doctype_facet_hier', 'title', 'pub_raw', 'property',
        'author', 'email', 'orcid', 'keyword', 'author_norm',
        'cite_read_boost', 'database', 'classic_factor', 'ack', 'page',
        'first_author', 'reader', 'read_count', 'indexstamp', 'issue', 'keyword_facet',
        'aff', 'facility', 'simbid']

ROWS = 2000
DB_PATH = os.path.expanduser("data/kpub.db")

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
# Array-valued fields stored as JSON; scalar fields stored as-is
ARRAY_FIELDS = {
    'links_data', 'citation', 'doi', 'keyword_schema', 'data', 'data_facet',
    'identifier', 'keyword_norm', 'reference', 'alternate_bibcode', 'arxiv_class',
    'reader', 'doctype_facet_hier', 'title', 'property', 'author', 'email',
    'orcid', 'keyword', 'author_norm', 'database', 'page', 'keyword_facet',
    'aff', 'facility', 'simbid',
}

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
con = sqlite3.connect(DB_PATH)
cur = con.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS test_publications (
        bibcode         TEXT PRIMARY KEY,
        date            TEXT,
        pub             TEXT,
        id              TEXT,
        volume          TEXT,
        links_data      TEXT,
        citation        TEXT,
        doi             TEXT,
        eid             TEXT,
        keyword_schema  TEXT,
        citation_count  INTEGER,
        data            TEXT,
        data_facet      TEXT,
        year            TEXT,
        identifier      TEXT,
        keyword_norm    TEXT,
        reference       TEXT,
        abstract        TEXT,
        recid           INTEGER,
        alternate_bibcode TEXT,
        arxiv_class     TEXT,
        first_author_norm TEXT,
        pubdate         TEXT,
        reader          TEXT,
        doctype         TEXT,
        doctype_facet_hier TEXT,
        title           TEXT,
        pub_raw         TEXT,
        property        TEXT,
        author          TEXT,
        email           TEXT,
        orcid           TEXT,
        keyword         TEXT,
        author_norm     TEXT,
        cite_read_boost REAL,
        database        TEXT,
        classic_factor  REAL,
        ack             TEXT,
        page            TEXT,
        first_author    TEXT,
        read_count      INTEGER,
        indexstamp      TEXT,
        issue           TEXT,
        keyword_facet   TEXT,
        aff             TEXT,
        facility        TEXT,
        simbid          TEXT
    )
""")

COLUMNS = [
    'bibcode', 'date', 'pub', 'id', 'volume', 'links_data', 'citation', 'doi',
    'eid', 'keyword_schema', 'citation_count', 'data', 'data_facet', 'year',
    'identifier', 'keyword_norm', 'reference', 'abstract', 'recid',
    'alternate_bibcode', 'arxiv_class', 'first_author_norm', 'pubdate', 'reader',
    'doctype', 'doctype_facet_hier', 'title', 'pub_raw', 'property', 'author',
    'email', 'orcid', 'keyword', 'author_norm', 'cite_read_boost', 'database',
    'classic_factor', 'ack', 'page', 'first_author', 'read_count', 'indexstamp',
    'issue', 'keyword_facet', 'aff', 'facility', 'simbid',
]

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
        f"INSERT OR REPLACE INTO test_publications ({col_list}) VALUES ({placeholders})",
        values,
    )
con.commit()
con.close()
print(f"Wrote {len(docs)} records to {DB_PATH}")