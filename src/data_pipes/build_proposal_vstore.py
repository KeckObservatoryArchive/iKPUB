"""
Build a postgres vector store of proposal coversheets from data/coversheets/*.json.

Reads cleaned JSON files, generates embeddings from ProgramTitle + ProgramSummary,
and inserts into a pgvector-enabled PostgreSQL table.

Usage:
    python -m src.data_pipes.build_proposal_vstore
"""

import json
import os
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).parents[2]
COVERSHEETS_DIR = PROJECT_ROOT / "data" / "coversheets"
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "all-mpnet-base-v2"
DB_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/kpub")
TABLE = "proposals"

EMBEDDING_DIM = 768  # all-mpnet-base-v2 output dimension

# Schema definition: json_key -> (pg_column, pg_type)
# Add/remove fields here — CREATE TABLE, INSERT, and param extraction all adapt.
FIELDS = {
    "Semester":        ("semester",        "TEXT"),
    "BaseInstrument":  ("base_instrument", "TEXT"),
    "Instrument":      ("instrument",      "TEXT"),
    "Institution":     ("institution",     "TEXT"),
    "ObsDate":         ("obs_date",        "TEXT"),
    "ObsIds":          ("obs_ids",         "TEXT[]"),
    "ObsLastNames":    ("obs_last_names",  "TEXT[]"),
    "PiFirstName":     ("pi_first_name",   "TEXT"),
    "PiLastName":      ("pi_last_name",    "TEXT"),
    "ProgramTitle":    ("program_title",   "TEXT"),
    "ProgramSummary":  ("program_summary", "TEXT"),
}

# Derived
PG_COLUMNS = [col for col, _ in FIELDS.values()]
PG_TYPES = {col: typ for col, typ in FIELDS.values()}
JSON_TO_PG = {json_key: col for json_key, (col, _) in FIELDS.items()}


def load_proposals(coversheets_dir: Path) -> list[dict]:
    """Load all proposal JSON files and return a flat list of entries."""
    proposals = []
    for path in sorted(coversheets_dir.glob("coversheets_*.json")):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            proposals.extend(data.values())
        elif isinstance(data, list):
            proposals.extend(data)
    return proposals


def embed_proposals(proposals: list[dict], model_path: Path) -> list[list[float]]:
    """Generate embeddings for each proposal's title + summary."""
    model = SentenceTransformer(str(model_path))
    texts = [
        f"{p.get('ProgramTitle', '')} {p.get('ProgramSummary', '')}".strip()
        for p in proposals
    ]
    return model.encode(texts, show_progress_bar=True).tolist()


def build_create_table() -> str:
    col_defs = ",\n    ".join(f"{col} {typ}" for col, typ in zip(PG_COLUMNS, PG_TYPES.values()))
    return f"""
    CREATE TABLE IF NOT EXISTS {TABLE} (
        id SERIAL PRIMARY KEY,
        {col_defs},
        embedding vector({EMBEDDING_DIM})
    );
    """


def build_insert() -> str:
    cols = PG_COLUMNS + ["embedding"]
    placeholders = ", ".join(f"%({c})s" for c in cols)
    col_list = ", ".join(cols)
    return f"INSERT INTO {TABLE} ({col_list}) VALUES ({placeholders})"


def extract_row(proposal: dict, embedding: list[float]) -> dict:
    """Map a proposal JSON object + embedding to a row dict keyed by pg column names."""
    row = {}
    for json_key, pg_col in JSON_TO_PG.items():
        val = proposal.get(json_key)
        # Coerce non-list values to list for TEXT[] columns
        if PG_TYPES[pg_col].endswith("[]") and not isinstance(val, list):
            val = [val] if val is not None else None
        row[pg_col] = val
    row["embedding"] = embedding
    return row


def insert_proposals(conn, proposals: list[dict], embeddings: list[list[float]]):
    insert_sql = build_insert()
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {TABLE}")
        for proposal, emb in zip(proposals, embeddings):
            cur.execute(insert_sql, extract_row(proposal, emb))
    conn.commit()


def main():
    proposals = load_proposals(COVERSHEETS_DIR)
    print(f"Loaded {len(proposals)} proposals")

    embeddings = embed_proposals(proposals, MODEL_PATH)
    print(f"Generated {len(embeddings)} embeddings")

    with psycopg.connect(DB_URL) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        conn.execute(build_create_table())
        insert_proposals(conn, proposals, embeddings)

    print(f"Inserted {len(proposals)} proposals into {TABLE}")


if __name__ == "__main__":
    main()
