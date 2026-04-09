"""Run a sequence of model configs from config/queue/ and collect results.

Usage:
    python src/eval/run_queue.py
    python src/eval/run_queue.py --save
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

from eval.test_harness import (
    CONFIG_KEY_MAP, MODELS, SAVE_DIR, eval_model, write_results,
)

PROJECT_ROOT = Path(__file__).parents[2]
QUEUE_DIR = PROJECT_ROOT / "config" / "queue"


def load_queue(queue_dir: Path) -> list[tuple[str, dict, str]]:
    """Load all YAML files from queue_dir. Returns (model_name, config, filename) tuples."""
    queue = []
    for path in sorted(queue_dir.glob("*.yaml")):
        with open(path) as f:
            raw = yaml.safe_load(f)
        for model_name, config in raw.items():
            config = config or {}
            mapped = {CONFIG_KEY_MAP.get(k, k): v for k, v in config.items()}
            queue.append((model_name, mapped, path.name))
    return queue


def main():
    parser = argparse.ArgumentParser(description="Run queued model configs")
    parser.add_argument("--save", action="store_true", help="save each trained model")
    parser.add_argument("--queue-dir", type=Path, default=QUEUE_DIR, help="directory of queue YAML files")
    args = parser.parse_args()

    queue = load_queue(args.queue_dir)
    if not queue:
        print(f"No configs found in {args.queue_dir}")
        sys.exit(1)

    print(f"Running {len(queue)} configs from {args.queue_dir}\n")

    for i, (model_name, config, filename) in enumerate(queue, 1):
        print(f"[{i}/{len(queue)}] {filename} → {model_name}")
        try:
            model, used_config, y_test, predictions, duration = eval_model(
                model_name, config=config,
            )
            results, out_path = write_results(model_name, used_config, y_test, predictions, duration)

            acc = results["results"]["accuracy"]
            cm = results["results"]["confusion_matrix"]
            print(f"  Accuracy: {acc:.4f}  (tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']})  {duration:.1f}s")
            print(f"  → {out_path}")

            if args.save:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                save_path = SAVE_DIR / f"{model_name}_{timestamp}"
                model.save(save_path)
                print(f"  Model saved to {save_path}")
        except Exception as e:
            print(f"  FAILED: {e}")

        print()

    print("Queue complete.")


if __name__ == "__main__":
    main()
