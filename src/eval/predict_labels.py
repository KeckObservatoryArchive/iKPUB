"""Batch-predict publication labels.

Supports three tasks that run sequentially:

  1. **keck** — Transformer classifier. Writes ilabel, keck_score.
  2. **drp** — LLM classifier on keck-positive papers. Writes idrp, drp_reason.
  3. **koa** — LLM classifier. Writes ikoa, koa_reason.

Usage (SQLite):
    python -m eval.predict_labels 2024
    python -m eval.predict_labels 2024 --task drp
    python -m eval.predict_labels --task koa

Usage (MongoDB):
    python -m eval.predict_labels 2024 --mongo
    python -m eval.predict_labels 2024 --mongo --task drp
    python -m eval.predict_labels 2024 --mongo --task koa --collection test_articles
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.prepare import load_publications, load_publications_mongo
from models.transformer import TransformerClassifier
from models.llm import LLMClassifier

PROJECT_ROOT = Path(__file__).parents[2]

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_DB = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
DEFAULT_TRANSFORMER = PROJECT_ROOT / "data" / "models" / "trained" / "transformer_2026-04-06_151800"
DEFAULT_LLM_MODEL = "gemma4:26b"
DEFAULT_LLM_HOST = "http://localhost:11434"

KECK_LOWER = 0.6
KECK_UPPER = 0.7
DRP_LOWER = 0.3
DRP_UPPER = 0.5


def parse_year_arg(year_str: str) -> tuple[int, int]:
    """Parse a year or year range (e.g. '2023' or '2020-2024')."""
    if "-" in year_str:
        start, end = year_str.split("-", 1)
        return int(start), int(end)
    year = int(year_str)
    return year, year


def keck_ilabel(prob: float) -> str:
    if prob >= KECK_UPPER:
        return "keck"
    if prob <= KECK_LOWER:
        return "not keck"
    return "unknown"


def drp_ilabel(prob: float) -> str:
    if prob >= DRP_UPPER:
        return "drp"
    if prob <= DRP_LOWER:
        return "not drp"
    return "unknown"


# ── Keck task ────────────────────────────────────────────────────────

def run_keck(year_start: int, year_end: int, model_path: Path, db_path: Path):
    print(f"Loading transformer from {model_path}")
    model = TransformerClassifier.load(model_path)

    pubs = load_publications(db_path, query=(
        f"SELECT * FROM keck WHERE year >= {year_start} AND year <= {year_end}"
    ))
    print(f"Loaded {len(pubs)} publications for years {year_start}-{year_end}")
    if pubs.empty:
        print("No publications found. Exiting.")
        return

    probs = model.predict(pubs, return_proba=True)

    results = pd.DataFrame({
        "bibcode": pubs["bibcode"].values,
        "ilabel": [keck_ilabel(p) for p in probs],
        "keck_score": probs.values,
    })

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS keck_predictions")
        results.to_sql("keck_predictions", con, index=False)

    summary = results["ilabel"].value_counts()
    print(f"\nWrote {len(results)} rows to keck_predictions")
    print(summary.to_string())


# ── DRP task ─────────────────────────────────────────────────────────

def run_drp(year_start: int, year_end: int, db_path: Path,
            model_name: str = DEFAULT_LLM_MODEL, host: str = DEFAULT_LLM_HOST,
            limit: int | None = None):
    # Load only keck-positive papers
    with sqlite3.connect(db_path) as con:
        keck_bibcodes = pd.read_sql(
            "SELECT bibcode FROM keck_predictions WHERE ilabel = 'keck'", con
        )
    if keck_bibcodes.empty:
        print("No keck-positive papers found in keck_predictions. Run --task keck first.")
        return

    bibcode_list = keck_bibcodes["bibcode"].tolist()
    placeholders = ",".join("?" for _ in bibcode_list)
    pubs = load_publications(db_path, query=(
        f"SELECT * FROM keck WHERE bibcode IN ({placeholders}) AND year >= {year_start} AND year <= {year_end}"
    ), params=bibcode_list)
    if limit:
        pubs = pubs.head(limit)
    print(f"Loaded {len(pubs)} keck-positive publications for DRP classification ({year_start}-{year_end})")

    if pubs.empty:
        return

    classifier = LLMClassifier(
        model_name=model_name,
        host=host,
        table="drp",
        task="drp",
    )
    scores, reasons = classifier.predict_with_reasons(pubs)

    results = pd.DataFrame({
        "bibcode": pubs["bibcode"].values,
        "drp_label": [drp_ilabel(p) for p in scores],
        "drp_score": scores.values,
        "reason": reasons.values,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
    })

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS drp_predictions")
        results.to_sql("drp_predictions", con, index=False)

    summary = results["drp_label"].value_counts()
    print(f"\nWrote {len(results)} rows to drp_predictions")
    print(summary.to_string())


# ── KOA task ────────────────────────────────────────────────────────

def run_koa(db_path: Path, year_start: int | None = None, year_end: int | None = None,
            model_name: str = DEFAULT_LLM_MODEL, host: str = DEFAULT_LLM_HOST,
            limit: int | None = None):
    query = "SELECT * FROM koa"
    if year_start is not None:
        query += f" WHERE year >= {year_start} AND year <= {year_end}"
    pubs = load_publications(db_path, query=query)
    if limit:
        pubs = pubs.head(limit)
    print(f"Loaded {len(pubs)} publications from koa table")

    if pubs.empty:
        return

    classifier = LLMClassifier(
        model_name=model_name,
        host=host,
        table="koa",
        task="koa",
    )
    scores, reasons = classifier.predict_with_reasons(pubs)

    results = pd.DataFrame({
        "bibcode": pubs["bibcode"].values,
        "koa_label": ["koa" if p >= 0.5 else "not koa" for p in scores],
        "koa_score": scores.values,
        "reason": reasons.values,
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
    })

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS koa_predictions")
        results.to_sql("koa_predictions", con, index=False)

    summary = results["koa_label"].value_counts()
    print(f"\nWrote {len(results)} rows to koa_predictions")
    print(summary.to_string())


def eval_koa(db_path: Path):
    """Compare koa_predictions against keck_manual ground truth."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("""
            SELECT p.bibcode, p.koa_label, p.koa_score, k.keck_manual
            FROM koa_predictions p
            JOIN koa k ON p.bibcode = k.bibcode
            WHERE k.keck_manual IS NOT NULL
        """, con)

    if df.empty:
        print("No predictions with ground truth found.")
        return

    predicted = (df["koa_label"] == "koa").astype(int)
    actual = df["keck_manual"].astype(int)

    tp = ((predicted == 1) & (actual == 1)).sum()
    fp = ((predicted == 1) & (actual == 0)).sum()
    fn = ((predicted == 0) & (actual == 1)).sum()
    tn = ((predicted == 0) & (actual == 0)).sum()
    accuracy = (tp + tn) / len(df)

    print(f"\n{'='*50}")
    print(f"KOA Evaluation: {len(df)} papers with ground truth")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {tp/(tp+fp):.4f}  Recall: {tp/(tp+fn):.4f}")
    print(f"Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"{'='*50}")

    errors = df[predicted.values != actual.values].sort_values("koa_score")

    fps = errors[errors["koa_label"] == "koa"]
    print(f"\n--- False Positives ({len(fps)}): predicted KOA, actually not ---")
    for _, row in fps.iterrows():
        print(f"  {row['bibcode']}  score={row['koa_score']:.2f}")

    fns = errors[errors["koa_label"] == "not koa"]
    print(f"\n--- False Negatives ({len(fns)}): predicted not KOA, actually is ---")
    for _, row in fns.iterrows():
        print(f"  {row['bibcode']}  score={row['koa_score']:.2f}")


# ── Merge ────────────────────────────────────────────────────────────

def merge_predictions(db_path: Path, include_drp: bool = False):
    """Rebuild the ``predictions`` table from keck + prediction tables."""
    with sqlite3.connect(db_path) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}

        if "keck_predictions" not in tables:
            print("keck_predictions table not found — skipping merge.")
            return

        exclude = {"keck_manual", "aff", "embedding", "full", "authors_clean"}
        col_info = con.execute("PRAGMA table_info(keck)").fetchall()
        keep_cols = [c[1] for c in col_info if c[1] not in exclude]
        col_list = ", ".join(f"k.{c}" for c in keep_cols)

        drp_join = ""
        drp_cols = ""
        if include_drp and "drp_predictions" in tables:
            drp_join = "LEFT JOIN drp_predictions d ON k.bibcode = d.bibcode"
            drp_cols = ", COALESCE(d.drp_label, 'not keck') AS drp_label, d.drp_score, d.reason AS drp_reason"
        else:
            drp_cols = ", 'not keck' AS drp_label, NULL AS drp_score, NULL AS drp_reason"

        query = f"""
            SELECT {col_list}, kp.ilabel, kp.keck_score{drp_cols}
            FROM keck k
            JOIN keck_predictions kp ON k.bibcode = kp.bibcode
            {drp_join}
        """

        merged = pd.read_sql(query, con)
        con.execute("DROP TABLE IF EXISTS predictions")
        merged.to_sql("predictions", con, index=False)

    print(f"\nMerged {len(merged)} rows into predictions table")


def merge_koa_predictions(db_path: Path, year_start: int, year_end: int):
    """Update the predictions table with KOA labels for the given year range."""
    with sqlite3.connect(db_path) as con:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "predictions" not in tables:
            print("predictions table not found — skipping KOA merge.")
            return
        if "koa_predictions" not in tables:
            print("koa_predictions table not found — skipping KOA merge.")
            return

        # Ensure KOA columns exist
        existing_cols = {r[1] for r in con.execute("PRAGMA table_info(predictions)")}
        for col, col_type in [("koa_label", "TEXT"), ("koa_score", "REAL"), ("koa_reason", "TEXT")]:
            if col not in existing_cols:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {col} {col_type}")

        # Update rows that have KOA predictions
        con.execute("""
            UPDATE predictions
            SET koa_label = kp.koa_label,
                koa_score = kp.koa_score,
                koa_reason = kp.reason
            FROM koa_predictions kp
            WHERE predictions.bibcode = kp.bibcode
              AND predictions.year >= ?
              AND predictions.year <= ?
        """, (str(year_start), str(year_end)))

        # Set rows in year range that weren't in the KOA query
        con.execute("""
            UPDATE predictions
            SET koa_label = 'not koa',
                koa_score = NULL,
                koa_reason = 'not found in koa query'
            WHERE year >= ? AND year <= ?
              AND koa_label IS NULL
        """, (str(year_start), str(year_end)))

        koa_counts = pd.read_sql(
            "SELECT koa_label, COUNT(*) as n FROM predictions WHERE year >= ? AND year <= ? GROUP BY koa_label",
            con, params=(str(year_start), str(year_end)),
        )

    print(f"\nMerged KOA predictions into predictions table ({year_start}-{year_end})")
    print(koa_counts.to_string(index=False))


# ── MongoDB tasks ───────────────────────────────────────────────────

def run_keck_mongo(year_start, year_end, model_path, collection):
    from pymongo import UpdateOne

    print(f"Loading transformer from {model_path}")
    model = TransformerClassifier.load(model_path)

    pubs = load_publications_mongo(collection, year_start, year_end)
    print(f"Loaded {len(pubs)} publications for years {year_start}-{year_end}")
    if pubs.empty:
        print("No publications found. Exiting.")
        return

    probs = model.predict(pubs, return_proba=True)

    ops = []
    for bibcode, prob in zip(pubs["bibcode"], probs):
        ops.append(UpdateOne(
            {"bibcode": bibcode},
            {"$set": {"ilabel": keck_ilabel(prob), "keck_score": float(prob)}},
        ))
    collection.bulk_write(ops)

    labels = [keck_ilabel(p) for p in probs]
    summary = pd.Series(labels).value_counts()
    print(f"\nUpdated {len(ops)} documents with keck predictions")
    print(summary.to_string())


def run_drp_mongo(year_start, year_end, collection,
                  model_name=DEFAULT_LLM_MODEL, host=DEFAULT_LLM_HOST,
                  limit=None):
    from pymongo import UpdateOne

    pubs = load_publications_mongo(collection, year_start, year_end)
    pubs = pubs[pubs["ilabel"] == "keck"]
    if limit:
        pubs = pubs.head(limit)
    print(f"Loaded {len(pubs)} keck-positive publications for DRP classification ({year_start}-{year_end})")

    if pubs.empty:
        return

    classifier = LLMClassifier(
        model_name=model_name, host=host, table="drp", task="drp",
    )
    scores, reasons = classifier.predict_with_reasons(pubs)

    ops = []
    for bibcode, score, reason in zip(pubs["bibcode"], scores, reasons):
        ops.append(UpdateOne(
            {"bibcode": bibcode},
            {"$set": {"idrp": drp_ilabel(score), "drp_reason": reason}},
        ))
    collection.bulk_write(ops)

    labels = [drp_ilabel(s) for s in scores]
    summary = pd.Series(labels).value_counts()
    print(f"\nUpdated {len(ops)} documents with DRP predictions")
    print(summary.to_string())


def run_koa_mongo(year_start, year_end, collection,
                  model_name=DEFAULT_LLM_MODEL, host=DEFAULT_LLM_HOST,
                  limit=None):
    from pymongo import UpdateOne

    pubs = load_publications_mongo(collection, year_start, year_end)
    if limit:
        pubs = pubs.head(limit)
    print(f"Loaded {len(pubs)} publications for KOA classification")

    if pubs.empty:
        return

    classifier = LLMClassifier(
        model_name=model_name, host=host, table="koa", task="koa",
    )
    scores, reasons = classifier.predict_with_reasons(pubs)

    ops = []
    for bibcode, score, reason in zip(pubs["bibcode"], scores, reasons):
        label = "koa" if score >= 0.5 else "not koa"
        ops.append(UpdateOne(
            {"bibcode": bibcode},
            {"$set": {"ikoa": label, "koa_reason": reason}},
        ))
    collection.bulk_write(ops)

    labels = ["koa" if s >= 0.5 else "not koa" for s in scores]
    summary = pd.Series(labels).value_counts()
    print(f"\nUpdated {len(ops)} documents with KOA predictions")
    print(summary.to_string())


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict publication labels.")
    parser.add_argument("year", nargs="?",
                        help="Single year (2023) or range (2020-2024). Required for keck/drp tasks.")
    parser.add_argument("--task", choices=["keck", "drp", "koa"], default="keck",
                        help="Classification task (default: keck)")
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate predictions against ground truth (koa task only)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TRANSFORMER,
                        help="Path to trained transformer model (keck task only)")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="Ollama model name (drp/koa tasks)")
    parser.add_argument("--llm-host", type=str, default=DEFAULT_LLM_HOST,
                        help="Ollama host URL (drp/koa tasks)")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--mongo", action="store_true", help="use MongoDB instead of SQLite")
    parser.add_argument("--collection", default="test_articles", help="MongoDB collection (default: test_articles)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max publications to classify (for quick iteration)")
    args = parser.parse_args()

    if args.year is None:
        parser.error("year is required")
    year_start, year_end = parse_year_arg(args.year)

    if args.mongo:
        from data.db_mongo_conn import from_env
        conn = from_env("kpub", args.collection)
        collection = conn.collection

        if args.task == "keck":
            run_keck_mongo(year_start, year_end, args.model_path, collection)
        elif args.task == "drp":
            run_drp_mongo(year_start, year_end, collection,
                          model_name=args.llm_model, host=args.llm_host,
                          limit=args.limit)
        elif args.task == "koa":
            run_koa_mongo(year_start, year_end, collection,
                          model_name=args.llm_model, host=args.llm_host,
                          limit=args.limit)
    else:
        if args.task == "keck":
            run_keck(year_start, year_end, args.model_path, args.db_path)
            merge_predictions(args.db_path)
        elif args.task == "drp":
            run_drp(year_start, year_end, args.db_path,
                    model_name=args.llm_model, host=args.llm_host,
                    limit=args.limit)
            merge_predictions(args.db_path, include_drp=True)
        elif args.task == "koa":
            if args.eval:
                eval_koa(args.db_path)
            else:
                run_koa(args.db_path, year_start, year_end,
                        model_name=args.llm_model, host=args.llm_host,
                        limit=args.limit)
                merge_koa_predictions(args.db_path, year_start, year_end)


if __name__ == "__main__":
    main()
