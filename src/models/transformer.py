"""
Train a publication classifier by fine-tuning RoBERTa with an MLP
classification head.

The full model (backbone + head) is fine-tuned end-to-end.
"""

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_kpub_classifier import KPUBClassifier
from .embedding import compose_text

PROJECT_ROOT = Path(__file__).parents[2]
ROBERTA_PATH = PROJECT_ROOT / "data" / "models" / "roberta-base"


def ensure_roberta(model_path: Path = ROBERTA_PATH) -> Path:
    """Download roberta-base to data/models/ if not already present."""
    if model_path.exists() and any(model_path.iterdir()):
        return model_path
    print(f"Downloading roberta-base to {model_path} ...")
    model_path.mkdir(parents=True, exist_ok=True)
    snapshot_download("roberta-base", local_dir=str(model_path))
    print("Download complete.")
    return model_path


# ---------------------------------------------------------------------------
# Custom MLP head (replaces HF's default single Linear layer)
# ---------------------------------------------------------------------------

class _ClassificationHead(nn.Module):
    """Replacement for RobertaClassificationHead. Takes the full sequence
    hidden states, extracts the [CLS] token, then runs through an MLP: 768 → 256 → 1."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, 256)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = features[:, 0, :]  # [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# ---------------------------------------------------------------------------
# Classifier (KPUBClassifier interface)
# ---------------------------------------------------------------------------

class TransformerClassifier(KPUBClassifier):
    """Fine-tuned RoBERTa + MLP for publication classification."""

    def __init__(
        self,
        model_path: str | Path = ROBERTA_PATH,
        max_length: int = 512,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 16,
        dropout: float = 0.3,
        max_samples: int | None = None,
        freeze_backbone: bool = False,
        device: str | None = None,
    ):
        self.model_path = str(model_path)
        self.max_length = max_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_samples = max_samples
        self.freeze_backbone = freeze_backbone
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = None
        self._model = None

    # -- internal helpers ---------------------------------------------------

    def _load_model(self):
        """Load tokenizer and model, swapping in our custom MLP head."""
        ensure_roberta(Path(self.model_path))

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        if self._model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path, num_labels=1, ignore_mismatched_sizes=True,
            )
            hidden_dim = model.config.hidden_size
            model.classifier = _ClassificationHead(input_dim=hidden_dim, dropout=self.dropout)
            if self.freeze_backbone:
                for param in model.roberta.parameters():
                    param.requires_grad = False
            self._model = model.to(self.device)

    def _tokenize(self, X: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Compose text from features and tokenize in batches."""
        texts = [compose_text(row) for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing text")]

        all_input_ids = []
        all_attention_masks = []
        tok_batch = 64
        for i in tqdm(range(0, len(texts), tok_batch), desc="Tokenizing"):
            batch = texts[i : i + tok_batch]
            encoded = self._tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            all_input_ids.append(encoded["input_ids"])
            all_attention_masks.append(encoded["attention_mask"])

        return {
            "input_ids": torch.cat(all_input_ids),
            "attention_mask": torch.cat(all_attention_masks),
        }

    # -- public API ---------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Tokenize training data and fine-tune RoBERTa + MLP."""
        self._load_model()

        if self.max_samples is not None and len(X_train) > self.max_samples:
            X_train = X_train.sample(n=self.max_samples, random_state=42)
            y_train = y_train.loc[X_train.index]
            print(f"  Subsampled to {self.max_samples} training examples")

        encoded = self._tokenize(X_train)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(input_ids, attention_mask, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        num_steps = len(loader) * self.epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps,
        )
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for ids_batch, mask_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self._model(input_ids=ids_batch, attention_mask=mask_batch)
                loss = criterion(outputs.logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * ids_batch.size(0)

            avg_loss = epoch_loss / len(dataset)
            print(f"  epoch {epoch + 1:>3}/{self.epochs}  loss={avg_loss:.4f}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Tokenize test data and classify with the fine-tuned model."""
        if self._model is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

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
                batch_preds = (torch.sigmoid(outputs.logits) >= 0.5).int().squeeze(1)
                preds_list.append(batch_preds.cpu())

        preds = torch.cat(preds_list).numpy()
        return pd.Series(preds, index=X_test.index)
