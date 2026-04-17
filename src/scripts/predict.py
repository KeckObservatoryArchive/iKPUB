"""Batch-predict publication labels.

Supports three tasks:

  1. **keck** — Transformer classifier. Writes ilabel, keck_score.
  2. **drp** — LLM classifier on keck-positive papers. Writes idrp, drp_reason.
  3. **koa** — LLM classifier. Writes ikoa, koa_reason.

Usage:
    python -m scripts.predict 2024
    python -m scripts.predict 2024 --task drp
    python -m scripts.predict 2024 --task koa --collection test_articles
"""

import argparse
from pathlib import Path

import pandas as pd
from pymongo import UpdateOne

from data.db_mongo_conn import from_env
from data.load_pubs import load_pubs
from models.transformer import TransformerClassifier
from models.llm import LLMClassifier

PROJECT_ROOT = Path(__file__).parents[2]

DEFAULT_TRANSFORMER = PROJECT_ROOT / "data" / "models" / "trained" / "transformer_2026-04-14_111534"
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


def run_keck(year_start, year_end, model_path, collection):
    print(f"Loading transformer from {model_path}")
    model = TransformerClassifier.load(model_path)

    pubs = load_pubs(collection, year_start, year_end)
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


def run_drp(year_start, year_end, collection,
            model_name=DEFAULT_LLM_MODEL, host=DEFAULT_LLM_HOST, limit=None):
    pubs = load_pubs(collection, year_start, year_end)
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


def run_koa(year_start, year_end, collection,
            model_name=DEFAULT_LLM_MODEL, host=DEFAULT_LLM_HOST, limit=None):
    pubs = load_pubs(collection, year_start, year_end)
    pubs = pubs[pubs["ilabel"] == "keck"]
    if limit:
        pubs = pubs.head(limit)
    print(f"Loaded {len(pubs)} keck-positive publications for KOA classification ({year_start}-{year_end})")

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


def main():
    parser = argparse.ArgumentParser(description="Predict publication labels.")
    parser.add_argument("year", nargs="?",
                        help="Single year (2023) or range (2020-2024).")
    parser.add_argument("--task", choices=["keck", "drp", "koa"], default="keck",
                        help="Classification task (default: keck)")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_TRANSFORMER,
                        help="Path to trained transformer model (keck task only)")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                        help="Ollama model name (drp/koa tasks)")
    parser.add_argument("--llm-host", type=str, default=DEFAULT_LLM_HOST,
                        help="Ollama host URL (drp/koa tasks)")
    parser.add_argument("--collection", default="test_articles",
                        help="MongoDB collection (default: test_articles)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max publications to classify (for quick iteration)")
    args = parser.parse_args()

    if args.year is None:
        parser.error("year is required")
    year_start, year_end = parse_year_arg(args.year)

    conn = from_env("kpub", args.collection)
    collection = conn.collection

    if args.task == "keck":
        run_keck(year_start, year_end, args.model_path, collection)
    elif args.task == "drp":
        run_drp(year_start, year_end, collection,
                model_name=args.llm_model, host=args.llm_host, limit=args.limit)
    elif args.task == "koa":
        run_koa(year_start, year_end, collection,
                model_name=args.llm_model, host=args.llm_host, limit=args.limit)

    del conn


if __name__ == "__main__":
    main()
