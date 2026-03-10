"""
Classify publications as directly resulting from an observation based on similarity with proposals
"""

from abc import abstractmethod

import pandas as pd

from .base_kpub_classifier import KPUBClassifier

EMBEDDING_ROWS = [...]

class ProposalMatcher(KPUBClassifier):
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the classifier on training data."""
        return

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Return predicted boolean labels for X_test."""
        embeddings = self._build_embeddings(X_test)
        proposals = self._get_proposals(year = ...)
        