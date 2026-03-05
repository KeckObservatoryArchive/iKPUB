# Standard Library
from pathlib import Path

# 3rd Party 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Local
from data_pipes.preprocess import load_publications, embed
from models.auto import AutoClassifier

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"

def eval_model(model=None):
    if model is None:
        model = AutoClassifier()

    pubs = load_publications(DB_PATH)
    X = pubs.drop("keck_manual", axis=1)
    y = pubs["keck_manual"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    return classification_report(y_test, predictions, output_dict=True, zero_division=0), y_test, predictions


def summary_statistics(report: dict, y_test, predictions) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    total = tn + fp + fn + tp
    return {
        "accuracy": report["accuracy"],
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "tn_pct": tn / total, "fp_pct": fp / total,
            "fn_pct": fn / total, "tp_pct": tp / total,
        },
    }


if __name__ == "__main__":
    report, y_test, predictions = eval_model()
    stats = summary_statistics(report, y_test, predictions)
    cm = stats.pop("confusion_matrix")
    print(f"accuracy: {stats['accuracy']:.4f}")
    print(f"confusion matrix: tn={cm['tn']} ({cm['tn_pct']:.1%})  fp={cm['fp']} ({cm['fp_pct']:.1%})  fn={cm['fn']} ({cm['fn_pct']:.1%})  tp={cm['tp']} ({cm['tp_pct']:.1%})")
