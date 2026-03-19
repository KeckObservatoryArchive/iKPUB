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
from src.data_pipes.prepare import load_publications
from models.transformer import TransformerClassifier
from models.embedding import EmbeddingClassifier
from models.auto import AutoClassifier

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
CONFIG_PATH = PROJECT_ROOT / "config" / "models.yaml"
OUTPUT_DIR = PROJECT_ROOT / "out" / "experiments"

MODELS = {
    "transformer": TransformerClassifier,
    "embedding": EmbeddingClassifier,
    "auto": AutoClassifier,
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


def build_model(model_name: str, config: dict | None = None):
    """Instantiate a classifier from its name and config dict (or YAML fallback)."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {', '.join(MODELS)}")
    if config is None:
        config = load_config(model_name)
    return MODELS[model_name](**config), config


SAVE_DIR = PROJECT_ROOT / "data" / "models" / "trained"


def eval_model(model_name: str, load_path: str | None = None, finetune_path: str | None = None,
               config: dict | None = None):
    pubs = load_publications(DB_PATH, query="SELECT * FROM publications WHERE year < 2024 and year > 1999")
    X = pubs.drop("keck_manual", axis=1)
    y = pubs["keck_manual"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if load_path is not None:
        model = TransformerClassifier.load(load_path)
        if config is None:
            config = load_config(model_name)
        start = time.time()
        predictions = model.predict(X_test)
        duration = time.time() - start
    elif finetune_path is not None:
        if config is None:
            config = load_config(model_name)
        config["load_path"] = finetune_path
        model = MODELS[model_name](**config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        duration = time.time() - start
    else:
        model, config = build_model(model_name, config=config)
        start = time.time()
        model.train(X_train, y_train)
        predictions = model.predict(X_test)
        duration = time.time() - start

    return model, config, y_test, predictions, duration


def write_results(model_name: str, config: dict, y_test, predictions, duration: float):
    """Write experiment results to out/experiments/."""
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_test, predictions)

    results = {
        "model": model_name,
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
    parser.add_argument("--save", action="store_true", help="save the trained model after evaluation")
    parser.add_argument("--load", metavar="PATH", help="load a saved model instead of training")
    parser.add_argument("--finetune", metavar="PATH", help="warm-start training from a saved model checkpoint")
    args = parser.parse_args()

    model, config, y_test, predictions, duration = eval_model(
        args.model, load_path=args.load, finetune_path=args.finetune,
    )
    results, out_path = write_results(args.model, config, y_test, predictions, duration)

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
