"""
Legacy transformer classifier for loading pre-AutoModel checkpoints.

These checkpoints used AutoModelForSequenceClassification with a custom
_ClassificationHead swapped onto the classifier attribute. This module
preserves that architecture for inference only — no training support.
"""

import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_kpub_classifier import KPUBClassifier, ensure_model
from .embedding import COMPOSE_FN

DEFAULT_HF_MODEL = "adsabs/astroBERT"


class _ClassificationHead(nn.Module):
    """Original classification head: hidden_dim → 256 → 1."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = features[:, 0, :] if features.dim() == 3 else features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def _get_classifier_attr(model: nn.Module) -> str:
    for name in ("classifier", "cls", "score"):
        if hasattr(model, name):
            return name
    raise ValueError(f"Could not find classifier head in {type(model).__name__}")


class LegacyTransformerClassifier(KPUBClassifier):
    """Load and run inference on pre-AutoModel transformer checkpoints."""

    def __init__(
        self,
        hf_model_name: str = DEFAULT_HF_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
        dropout: float = 0.3,
        extraction_mode: str = "sentence",
        table: str = "publications",
        device: str | None = None,
    ):
        self.hf_model_name = hf_model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.dropout = dropout
        self.extraction_mode = extraction_mode
        self.table = table
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self._tokenizer = None
        self._model = None
        self._temperature = 1.0

    def _load_model(self):
        model_path = ensure_model(self.hf_model_name)

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self._model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path, num_labels=1, ignore_mismatched_sizes=True,
            )
            hidden_dim = model.config.hidden_size
            cls_attr = _get_classifier_attr(model)
            setattr(model, cls_attr, _ClassificationHead(input_dim=hidden_dim, dropout=self.dropout))
            self._model = model.to(self.device)

    def _tokenize(self, X: pd.DataFrame) -> dict[str, torch.Tensor]:
        compose_fn = COMPOSE_FN[self.table]
        texts = [compose_fn(row, extraction_mode=self.extraction_mode) for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing text")]

        all_input_ids = []
        all_attention_masks = []
        tok_batch = 64
        for i in tqdm(range(0, len(texts), tok_batch), desc="Tokenizing"):
            batch = texts[i : i + tok_batch]
            encoded = self._tokenizer(
                batch, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])

        return {
            "input_ids": torch.cat(all_input_ids),
            "attention_mask": torch.cat(all_attention_masks),
        }

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        raise NotImplementedError("LegacyTransformerClassifier is inference-only. Use TransformerClassifier for training.")

    def predict(self, X_test: pd.DataFrame, return_proba: bool = False) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Model has not been loaded. Call load() first.")

        encoded = self._tokenize(X_test)
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self._model.eval()
        preds_list = []

        with torch.no_grad():
            for ids_batch, mask_batch in loader:
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                outputs = self._model(input_ids=ids_batch, attention_mask=mask_batch)
                probs = torch.sigmoid(outputs.logits / self._temperature).squeeze(1)
                if return_proba:
                    preds_list.append(probs.cpu())
                else:
                    preds_list.append((probs >= 0.5).int().cpu())

        preds = torch.cat(preds_list).numpy()
        return pd.Series(preds, index=X_test.index)

    @classmethod
    def load(cls, path: str | Path) -> "LegacyTransformerClassifier":
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        instance = cls(
            hf_model_name=meta["hf_model_name"],
            max_length=meta["max_length"],
            batch_size=meta["batch_size"],
            dropout=meta["dropout"],
            extraction_mode=meta.get("extraction_mode", "sentence"),
        )
        instance._load_model()
        instance._model.load_state_dict(torch.load(path / "model.pt", map_location=instance.device))
        instance._model.eval()
        instance._temperature = meta.get("temperature", 1.0)

        print(f"Legacy model loaded from {path} (T={instance._temperature:.4f})")
        return instance
