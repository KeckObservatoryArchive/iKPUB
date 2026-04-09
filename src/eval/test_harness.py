"""Train and evaluate KPUB classifiers.

Loads a table from kpub.db, splits into train/test, and reports accuracy
and confusion matrix. Supports training from scratch, loading a saved model
for inference, or finetuning from a checkpoint.

Usage:
    python src/eval/test_harness.py transformer --table koa
    python src/eval/test_harness.py embedding --table keck --save

When training on a table that contains rows from another table (e.g.
"combined" includes all of "koa"), use --holdout-table to define the test
set from that table's bibcodes. This ensures the same test split is used
across training stages and prevents data leakage:

    python src/eval/test_harness.py transformer --table combined --holdout-table koa --save
    python src/eval/test_harness.py transformer --table koa --finetune <path>
"""

# Standard Library
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# 3rd Party
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Local
from data.prepare import load_publications, load_manual_pubs
from models.transformer import TransformerClassifier
from models.embedding import EmbeddingClassifier
from models.snippet import SnippetClassifier

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
OUTPUT_DIR = PROJECT_ROOT / "out" / "experiments"

MODELS = {
    "transformer": TransformerClassifier,
    "embedding": EmbeddingClassifier,
    "snippet": SnippetClassifier,
}

# Map config keys to constructor param names where they differ
CONFIG_KEY_MAP = {
    "learning_rate": "lr",
}


def load_config(model_name: str) -> dict:
    """Load model config from models.yaml."""
    with open(CONFIG_PATH) as f:
        all_config = yaml.safe_load(f)
    config = all_config.get(model_name, {})
    # Remap keys to match constructor parameter names
    return {CONFIG_KEY_MAP.get(k, k): v for k, v in config.items()}


def build_model(model_name: str, table: str = "keck", config: dict | None = None):
    """Instantiate a classifier from its name and config dict (or YAML fallback)."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODELS)}")
    if config is None:
        config = load_config(model_name)
    config["table"] = table
    return MODELS[model_name](**config), config


SAVE_DIR = PROJECT_ROOT / "data" / "models" / "trained"


def _holdout_split(pubs, holdout_table: str):
    """Split using bibcodes from holdout_table to define a consistent test set."""
    holdout_pubs = load_publications(
        DB_PATH, query=f"SELECT bibcode, keck_manual FROM {holdout_table} WHERE year < 2024 and year > 1999",
    )
    _, test_bibcodes = train_test_split(holdout_pubs["bibcode"], test_size=0.2, random_state=42)
    test_set = set(test_bibcodes)

    is_test = pubs["bibcode"].isin(test_set)
    X = pubs.drop("keck_manual", axis=1)
    y = pubs["keck_manual"]
    return X[~is_test], X[is_test], y[~is_test], y[is_test]


def eval_model(model_name: str, table: str = "keck", load_path: str | None = None,
               finetune_path: str | None = None, config: dict | None = None,
               holdout_table: str | None = None, eval_table: str | None = None,
               eval_fraction: float = 1.0, eval_db: str | None = None,
               year_start: int = 2000, year_end: int = 2023):
    if eval_db is not None:
        eval_pubs = load_manual_pubs(eval_db, str(DB_PATH), table=table, year_start=year_start, year_end=year_end)
        X_test = eval_pubs.drop("keck_manual", axis=1)
        y_test = eval_pubs["keck_manual"]
        X_train, y_train = None, None
    else:
        pubs = load_publications(DB_PATH, query=f"SELECT * FROM {table} WHERE year <= {year_end} and year >= {year_start}")
        if eval_table is not None:
            eval_pubs = load_publications(DB_PATH, query=f"SELECT * FROM {eval_table} WHERE year <= {year_end} and year >= {year_start}")
            eval_pubs = eval_pubs[~eval_pubs["bibcode"].isin(pubs["bibcode"])]
            X_train = pubs.drop("keck_manual", axis=1)
            y_train = pubs["keck_manual"]
            X_test = eval_pubs.drop("keck_manual", axis=1)
            y_test = eval_pubs["keck_manual"]
        elif holdout_table is not None:
            X_train, X_test, y_train, y_test = _holdout_split(pubs, holdout_table)
        else:
            X = pubs.drop("keck_manual", axis=1)
            y = pubs["keck_manual"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if eval_fraction < 1.0:
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=eval_fraction, random_state=42
        )

    if load_path is not None:
        model = MODELS[model_name].load(load_path)
        if config is None:
            config = load_config(model_name)
        start = time.time()
        predictions = model.predict(X_test)
        duration = time.time() - start
    elif finetune_path is not None:
        if config is None:
            config = load_config(model_name)
        config["load_path"] = finetune_path
        config["table"] = table
        model = MODELS[model_name](**config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        duration = time.time() - start
    else:
        model, config = build_model(model_name, table=table, config=config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        duration = time.time() - start

    return model, config, y_test, predictions, duration


def write_results(model_name: str, table: str, config: dict, y_test, predictions, duration: float):
    """Write experiment results to out/experiments/."""
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_test, predictions)

    results = {
        "model": model_name,
        "table": table,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "results": {
            "accuracy": round(accuracy, 4),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            },
        },
        "duration_seconds": round(duration, 1),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{model_name}.jsonl"
    with open(out_path, "a") as f:
        f.write(json.dumps(results) + "\n")

    return results, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a KPUB classifier")
    parser.add_argument("model", choices=MODELS.keys(), help="model to evaluate")
    parser.add_argument("--table", default="keck", help="DB table to use (default: keck)")
    parser.add_argument("--save", action="store_true", help="save the trained model after evaluation")
    parser.add_argument("--load", metavar="PATH", help="load a saved model instead of training")
    parser.add_argument("--finetune", metavar="PATH", help="warm-start training from a saved model checkpoint")
    parser.add_argument("--holdout-table", metavar="TABLE",
                        help="define test set from this table's bibcodes to prevent train/test overlap across tables")
    parser.add_argument("--eval-table", metavar="TABLE",
                        help="evaluate on this table instead of splitting --table")
    parser.add_argument("--eval-fraction", type=float, default=1.0, metavar="FRAC",
                        help="fraction of eval/test data to use (e.g. 0.2 for 20%%)")
    parser.add_argument("--eval-db", metavar="PATH",
                        help="evaluate on data from this database (e.g. data/pubs/manual_kpub.db)")
    parser.add_argument("--year", metavar="RANGE", default="2000-2023",
                        help="year or year range, e.g. 2024 or 2020-2024 (default: 2000-2023)")
    args = parser.parse_args()

    if "-" in args.year:
        y_start, y_end = args.year.split("-", 1)
        year_start, year_end = int(y_start), int(y_end)
    else:
        year_start = year_end = int(args.year)

    model, config, y_test, predictions, duration = eval_model(
        args.model, table=args.table, load_path=args.load, finetune_path=args.finetune,
        holdout_table=args.holdout_table, eval_table=args.eval_table,
        eval_fraction=args.eval_fraction, eval_db=args.eval_db,
        year_start=year_start, year_end=year_end,
    )
    results, out_path = write_results(args.model, args.table, config, y_test, predictions, duration)

    if args.save:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_path = SAVE_DIR / f"{args.model}_{timestamp}"
        model.save(save_path)

    cm = results["results"]["confusion_matrix"]
    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"Accuracy: {results['results']['accuracy']:.4f}")
    print(f"Confusion matrix: tn={cm['tn']}  fp={cm['fp']}  fn={cm['fn']}  tp={cm['tp']}")
    print(f"Duration: {duration:.1f}s")
    print(f"Results saved to: {out_path}")
