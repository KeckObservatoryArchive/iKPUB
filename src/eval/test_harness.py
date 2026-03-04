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

    pubs = embed(load_publications(DB_PATH))
    X = pubs.drop("keck_manual", axis=1)
    y = pubs["keck_manual"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.train(X_train, y_train)

    predictions = model.predict(X_test)
    return classification_report(y_test, predictions, output_dict=True, zero_division=0), y_test, predictions


def summary_statistics(report: dict, y_test, predictions) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


if __name__ == "__main__":
    report, y_test, predictions = eval_model()
    stats = summary_statistics(report, y_test, predictions)
    cm = stats.pop("confusion_matrix")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}")
    print(f"confusion matrix: tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")
