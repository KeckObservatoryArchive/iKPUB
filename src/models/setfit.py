"""
SetFit-style classifier for few-shot publication classification.

Fine-tunes a sentence-transformer via contrastive learning on text pairs,
then trains a logistic regression classification head. Implemented directly
with sentence-transformers (no setfit package dependency).
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib

from .base_kpub_classifier import KPUBClassifier, ensure_model
from .embedding import COMPOSE_FN


def _generate_pairs(texts: list[str], labels: list[int], num_iterations: int,
                     oversample: bool = True) -> list[InputExample]:
    """Generate contrastive sentence pairs for training.

    When oversample=False, generates num_iterations pairs per example — so
    the majority class dominates.

    When oversample=True, generates a fixed budget of pairs per class
    (num_iterations × max_class_size), oversampling minority classes with
    replacement so each class contributes equally.
    """
    by_label: dict[int, list[str]] = {}
    for text, label in zip(texts, labels):
        by_label.setdefault(label, []).append(text)

    max_class_size = max(len(g) for g in by_label.values())

    pairs = []
    for label, group in by_label.items():
        other_labels = [l for l in by_label if l != label]
        other_texts = [t for l in other_labels for t in by_label[l]]

        if oversample:
            budget = num_iterations * max_class_size
            anchors = random.choices(group, k=budget)
        else:
            anchors = [t for t in group for _ in range(num_iterations)]

        for text in anchors:
            # Positive pair: same class
            partner = random.choice(group)
            pairs.append(InputExample(texts=[text, partner], label=1.0))
            # Negative pair: different class
            neg = random.choice(other_texts)
            pairs.append(InputExample(texts=[text, neg], label=0.0))

    random.shuffle(pairs)
    return pairs


class SetFitClassifier(KPUBClassifier):
    """Few-shot classifier using contrastive fine-tuning + logistic regression."""

    def __init__(
        self,
        hf_model_name: str = "adsabs/astroBERT",
        num_iterations: int = 20,
        epochs: int = 1,
        batch_size: int = 32,
        extraction_mode: str = "sentence",
        oversample: bool = True,
        table: str = "koa",
        device: str | None = None,
    ):
        self.hf_model_name = hf_model_name
        self.num_iterations = num_iterations
        self.epochs = epochs
        self.batch_size = batch_size
        self.extraction_mode = extraction_mode
        self.oversample = oversample
        self.table = table
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self._encoder: SentenceTransformer | None = None
        self._head: LogisticRegression | None = None

    def _compose_texts(self, X: pd.DataFrame) -> list[str]:
        compose_fn = COMPOSE_FN[self.table]
        return [compose_fn(row, extraction_mode=self.extraction_mode)
                for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing text")]

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        texts = self._compose_texts(X_train)
        labels = y_train.tolist()

        # Load encoder: construct from modules so any HF model works
        model_path = ensure_model(self.hf_model_name)
        word_model = models.Transformer(str(model_path))
        pooling = models.Pooling(word_model.get_word_embedding_dimension(), pooling_mode="mean")
        self._encoder = SentenceTransformer(modules=[word_model, pooling], device=self.device)

        # Phase 1: contrastive fine-tuning of the encoder
        pairs = _generate_pairs(texts, labels, self.num_iterations, self.oversample)
        train_loader = DataLoader(pairs, shuffle=True, batch_size=self.batch_size)
        loss_fn = losses.CosineSimilarityLoss(self._encoder)

        print(f"  Contrastive fine-tuning: {len(pairs)} pairs, {self.epochs} epoch(s)")
        self._encoder.fit(
            train_objectives=[(train_loader, loss_fn)],
            epochs=self.epochs,
            show_progress_bar=True,
        )

        # Phase 2: train logistic regression head on fine-tuned embeddings
        embeddings = self._encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        self._head = LogisticRegression(max_iter=500)
        self._head.fit(embeddings, labels)
        print(f"  Head trained on {len(texts)} examples")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._encoder is None or self._head is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        texts = self._compose_texts(X_test)
        embeddings = self._encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        preds = self._head.predict(embeddings)
        return pd.Series(preds.astype(int), index=X_test.index)

    def save(self, path: str | Path) -> Path:
        if self._encoder is None or self._head is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._encoder.save(str(path / "encoder"))
        joblib.dump(self._head, path / "head.joblib")

        meta = {
            "hf_model_name": self.hf_model_name,
            "num_iterations": self.num_iterations,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "extraction_mode": self.extraction_mode,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SetFitClassifier":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        instance = cls(
            hf_model_name=meta["hf_model_name"],
            num_iterations=meta["num_iterations"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            extraction_mode=meta.get("extraction_mode", "sentence"),
        )
        instance._encoder = SentenceTransformer(str(path / "encoder"))
        instance._head = joblib.load(path / "head.joblib")
        print(f"Model loaded from {path}")
        return instance
