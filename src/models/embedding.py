"""Train a publication classifier using an embedding model with a PyTorch
classification head.

Embeddings are built at train/predict time from raw publication features
(not from pre-computed DB embeddings).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base_kpub_classifier import KPUBClassifier, ensure_model
from .heads import DeepMLPHead
from data.compose import COMPOSE_FN, _extract_relevant_sentences

DEFAULT_HF_MODEL = "allenai/specter"


class EmbeddingClassifier(KPUBClassifier):
    """Sentence-transformer embeddings + PyTorch MLP classification head."""

    def __init__(
        self,
        hf_model_name: str = DEFAULT_HF_MODEL,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        dropout: float = 0.3,
        extraction_mode: str = "sentence",
        table: str = "keck",
        device: str | None = None,
    ):
        self.hf_model_name = hf_model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.extraction_mode = extraction_mode
        self.table = table
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self._encoder: SentenceTransformer | None = None
        self._head: DeepMLPHead | None = None

    # -- internal helpers ---------------------------------------------------

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            model_path = ensure_model(self.hf_model_name)
            self._encoder = SentenceTransformer(str(model_path))
        return self._encoder

    def _embed(self, X: pd.DataFrame) -> np.ndarray:
        """Compose text from features and encode with sentence-transformers."""
        compose_fn = COMPOSE_FN[self.table]

        def _compose_with_relevant_first(row):
            full = compose_fn(row, extraction_mode=self.extraction_mode)
            relevant = _extract_relevant_sentences(full, mode=self.extraction_mode)
            if relevant:
                return relevant + " " + full
            return full

        texts = [_compose_with_relevant_first(row) for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing text")]
        encoder = self._get_encoder()
        return encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # -- public API ---------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Embed training data and train the classification head."""
        embeddings = self._embed(X_train)
        embedding_dim = embeddings.shape[1]

        X_t = torch.tensor(embeddings, dtype=torch.float32)
        y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._head = DeepMLPHead(input_dim=embedding_dim, dropout=self.dropout)
        self._head.to(self.device)
        self._head.train()

        optimizer = torch.optim.Adam(self._head.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self._head(X_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * X_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"  epoch {epoch + 1:>3}/{self.epochs}  loss={avg_loss:.4f}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Embed test data and classify with the trained head."""
        if self._head is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        embeddings = self._embed(X_test)
        X_t = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        self._head.eval()
        with torch.no_grad():
            logits = self._head(X_t)
            preds = (torch.sigmoid(logits) >= 0.5).int().squeeze(1)

        return pd.Series(preds.cpu().numpy(), index=X_test.index)
