"""Train (and optionally evaluate) KPUB classifiers.

Loads publications from MongoDB, derives labels from the ``affiliation`` field
(``"keck"`` → positive, everything else → negative), splits into train/test by
default, and reports accuracy and confusion matrix. Supports training from
scratch, loading a saved model for inference, or finetuning from a checkpoint.
Use ``--no-test`` to train on all labeled data without holding out a test set.

Usage:
    python src/scripts/train.py transformer --save
    python src/scripts/train.py embedding --table koa
    python src/scripts/train.py transformer --no-test --save
    python src/scripts/train.py transformer --year 2020-2024 --collection test_articles
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
from data.db_mongo_conn import from_env
from data.load_pubs import load_pubs
from models.transformer import TransformerClassifier
from models.embedding import EmbeddingClassifier
from models.snippet import SnippetClassifier
from models.llm import LLMClassifier

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
OUTPUT_DIR = PROJECT_ROOT / "out" / "experiments"
SAVE_DIR = PROJECT_ROOT / "data" / "models" / "trained"

MODELS = {
    "transformer": TransformerClassifier,
    "embedding": EmbeddingClassifier,
    "snippet": SnippetClassifier,
    "llm": LLMClassifier,
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
    return {CONFIG_KEY_MAP.get(k, k): v for k, v in config.items()}


def build_model(model_name: str, table: str = "keck", config: dict | None = None):
    """Instantiate a classifier from its name and config dict (or YAML fallback)."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODELS)}")
    if config is None:
        config = load_config(model_name)
    config["table"] = table
    return MODELS[model_name](**config), config


def load_labeled_pubs(collection, year_start: int, year_end: int):
    """Load pubs from Mongo and derive binary ``keck_manual`` from affiliation.

    Returns (pubs, stats) where stats reports class balance and skipped rows.
    """
    pubs = load_pubs(collection, year_start, year_end)
    before = len(pubs)
    pubs = pubs[pubs["affiliation"].notna() & (pubs["affiliation"] != "")]
    skipped = before - len(pubs)
    pubs["keck_manual"] = (pubs["affiliation"] == "keck").astype(int)
    stats = {
        "skipped_no_affiliation": skipped,
        "n_positives": int((pubs["keck_manual"] == 1).sum()),
        "n_negatives": int((pubs["keck_manual"] == 0).sum()),
    }
    return pubs, stats


def run(model_name: str, collection, table: str = "keck",
        load_path: str | None = None, finetune_path: str | None = None,
        config: dict | None = None, eval_fraction: float = 1.0,
        year_start: int = 2000, year_end: int = 2023, no_test: bool = False):
    pubs, stats = load_labeled_pubs(collection, year_start, year_end)

    X = pubs.drop("keck_manual", axis=1)
    y = pubs["keck_manual"]

    if no_test:
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
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
        predictions = None if no_test else model.predict(X_test)
        duration = time.time() - start
    elif finetune_path is not None:
        if config is None:
            config = load_config(model_name)
        config["load_path"] = finetune_path
        config["table"] = table
        model = MODELS[model_name](**config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = None if no_test else model.predict(X_test)
        duration = time.time() - start
    else:
        model, config = build_model(model_name, table=table, config=config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = None if no_test else model.predict(X_test)
        duration = time.time() - start

    return model, config, y_test, predictions, duration, stats


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
    parser = argparse.ArgumentParser(description="Train a KPUB classifier")
    parser.add_argument("model", choices=MODELS.keys(), help="model to train")
    parser.add_argument("--table", default="keck",
                        help="classification task / text-composition mode (default: keck)")
    parser.add_argument("--save", action="store_true", help="save the trained model after evaluation")
    parser.add_argument("--load", metavar="PATH", help="load a saved model instead of training")
    parser.add_argument("--finetune", metavar="PATH", help="warm-start training from a saved model checkpoint")
    parser.add_argument("--eval-fraction", type=float, default=1.0, metavar="FRAC",
                        help="fraction of eval/test data to use (e.g. 0.2 for 20%%)")
    parser.add_argument("--no-test", action="store_true",
                        help="train on all labeled data, skip holdout eval and result-writing")
    parser.add_argument("--year", metavar="RANGE", default="2000-2023",
                        help="year or year range, e.g. 2024 or 2020-2024 (default: 2000-2023)")
    parser.add_argument("--collection", default="test_articles",
                        help="MongoDB collection (default: test_articles)")
    args = parser.parse_args()

    if args.no_test and args.load:
        parser.error("--no-test cannot be combined with --load (nothing to train)")
    if args.no_test and args.eval_fraction != 1.0:
        parser.error("--no-test cannot be combined with --eval-fraction")

    if "-" in args.year:
        y_start, y_end = args.year.split("-", 1)
        year_start, year_end = int(y_start), int(y_end)
    else:
        year_start = year_end = int(args.year)

    mongo_conn = from_env("kpub", args.collection)

    model, config, y_test, predictions, duration, stats = run(
        args.model, mongo_conn.collection, table=args.table,
        load_path=args.load, finetune_path=args.finetune,
        eval_fraction=args.eval_fraction,
        year_start=year_start, year_end=year_end,
        no_test=args.no_test,
    )

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")

    if args.no_test:
        print(f"Trained on all labeled data (no holdout)")
        print(f"Duration: {duration:.1f}s")
    else:
        results, out_path = write_results(args.model, args.table, config, y_test, predictions, duration)
        cm = results["results"]["confusion_matrix"]
        print(f"Accuracy: {results['results']['accuracy']:.4f}")
        print(f"Confusion matrix: tn={cm['tn']}  fp={cm['fp']}  fn={cm['fn']}  tp={cm['tp']}")
        print(f"Duration: {duration:.1f}s")

    print(f"Positives (affiliation=keck): {stats['n_positives']}")
    print(f"Negatives: {stats['n_negatives']}")
    print(f"Skipped (no affiliation): {stats['skipped_no_affiliation']}")

    if args.save:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save_path = SAVE_DIR / f"{args.model}_{timestamp}"
        model.save(save_path)
        print(f"Model saved to: {save_path}")

    if not args.no_test:
        print(f"Results saved to: {out_path}")

    del mongo_conn
