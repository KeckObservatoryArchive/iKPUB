"""
Abstract base class for all KPUB publication classifiers.

Subclasses must implement `train` and `predict`. The expected input is a
DataFrame of embeddings (one row per publication) and a boolean Series of
labels (True = Keck publication).
"""
from abc import ABC, abstractmethod
import pandas as pd

class KPUBClassifier(ABC):

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the classifier on training data."""
        ...

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Return predicted boolean labels for X_test."""
        ...