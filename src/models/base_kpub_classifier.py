"""
Abstract base class for all KPUB publication classifiers.

Subclasses must implement `train` and `predict`. The expected input is a
DataFrame of embeddings (one row per publication) and a boolean Series of
labels (True = Keck publication).
"""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def ensure_model(hf_model_name: str, models_dir: Path = MODELS_DIR) -> Path:
    """Download a HuggingFace model to data/models/ if not already present."""
    local_path = models_dir / hf_model_name
    if local_path.exists() and any(local_path.iterdir()):
        return local_path
    print(f"Downloading {hf_model_name} to {local_path} ...")
    local_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(hf_model_name, local_dir=str(local_path))
    print("Download complete.")
    return local_path

class KPUBClassifier(ABC):

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the classifier on training data."""
        ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Return predicted boolean labels for X_test."""
        ...