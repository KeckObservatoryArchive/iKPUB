"""
Train a publication classifier by fine-tuning a HuggingFace transformer
with an MLP classification head.

The full model (backbone + head) is fine-tuned end-to-end.
"""

import copy
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .base_kpub_classifier import KPUBClassifier, ensure_model
from .embedding import COMPOSE_FN

DEFAULT_HF_MODEL = "adsabs/astroBERT"


# ---------------------------------------------------------------------------
# Custom MLP head (replaces HF's default single Linear layer)
# ---------------------------------------------------------------------------

class _ClassificationHead(nn.Module):
    """Custom classification head. Takes the full sequence hidden states,
    extracts the [CLS] token, then runs through an MLP: hidden_dim → 256 → 1."""

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


def _get_backbone(model: nn.Module) -> nn.Module:
    """Return the transformer backbone, regardless of architecture.

    HF names the backbone after the architecture: model.roberta, model.distilbert,
    model.bert, etc. The classifier head is always a separate attribute.
    """
    for name, child in model.named_children():
        if name != "classifier" and name != "pre_classifier":
            return child
    raise ValueError(f"Could not identify backbone in {type(model).__name__}")


def _get_classifier_attr(model: nn.Module) -> str:
    """Return the attribute name HF uses for the classification head.

    Most architectures use 'classifier'; some use 'cls' or 'score'.
    """
    for name in ("classifier", "cls", "score"):
        if hasattr(model, name):
            return name
    raise ValueError(f"Could not find classifier head in {type(model).__name__}")


# ---------------------------------------------------------------------------
# Focal loss (Lin et al. 2017)
# ---------------------------------------------------------------------------

def _focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    """Binary focal loss from raw logits.

    Reduces the contribution of well-classified (confident) examples so
    training focuses on hard/misclassified ones.  When *gamma* = 0 this
    is identical to ``BCEWithLogitsLoss(reduction='mean')``.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    # p_t = probability assigned to the true class
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ---------------------------------------------------------------------------
# Classifier (KPUBClassifier interface)
# ---------------------------------------------------------------------------

class TransformerClassifier(KPUBClassifier):
    """Fine-tuned transformer + MLP for publication classification."""

    def __init__(
        self,
        hf_model_name: str = DEFAULT_HF_MODEL,
        max_length: int = 512,
        epochs: int = 5,
        lr: float = 2e-5,
        batch_size: int = 16,
        dropout: float = 0.3,
        max_samples: int | None = None,
        freeze_backbone: bool = False,
        extraction_mode: str = "sentence",
        table: str = "publications",
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.0,
        patience: int | None = None,
        focal_loss_gamma: float = 0.0,
        load_path: str | None = None,
        device: str | None = None,
    ):
        self.hf_model_name = hf_model_name
        self.max_length = max_length
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.max_samples = max_samples
        self.freeze_backbone = freeze_backbone
        self.extraction_mode = extraction_mode
        self.table = table
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.focal_loss_gamma = focal_loss_gamma
        self.load_path = load_path
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self._tokenizer = None
        self._model = None
        self._temperature = 1.0  # learned via post-hoc calibration

    # -- internal helpers ---------------------------------------------------

    def _load_model(self):
        """Load tokenizer and model, swapping in our custom MLP head."""
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
            if self.load_path is not None:
                state_dict = torch.load(
                    Path(self.load_path) / "model.pt", map_location=self.device,
                )
                model.load_state_dict(state_dict)
                print(f"  Warm-start: loaded weights from {self.load_path}")
            if self.freeze_backbone:
                backbone = _get_backbone(model)
                for param in backbone.parameters():
                    param.requires_grad = False
            self._model = model.to(self.device)

    def _tokenize(self, X: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Compose text from features and tokenize in batches."""
        compose_fn = COMPOSE_FN[self.table]
        texts = [compose_fn(row, extraction_mode=self.extraction_mode) for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing text")]

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
        """Tokenize training data and fine-tune the transformer + MLP."""
        self._load_model()

        if self.max_samples is not None and len(X_train) > self.max_samples:
            X_train = X_train.sample(n=self.max_samples, random_state=42)
            y_train = y_train.loc[X_train.index]
            print(f"  Subsampled to {self.max_samples} training examples")

        encoded = self._tokenize(X_train)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        # Split off a validation set for early stopping
        n = len(input_ids)
        n_val = max(1, int(n * 0.1))
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        train_dataset = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
        val_dataset = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self._model.train()

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        num_steps = len(loader) * self.epochs
        num_warmup = int(num_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_steps,
        )
        if self.focal_loss_gamma > 0:
            def criterion(logits, targets):
                return _focal_bce_with_logits(logits, targets, self.focal_loss_gamma)
            print(f"  Using focal loss (gamma={self.focal_loss_gamma})")
        else:
            criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            # -- training --
            self._model.train()
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

            avg_train_loss = epoch_loss / len(train_dataset)

            # -- validation --
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for ids_batch, mask_batch, y_batch in val_loader:
                    ids_batch = ids_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    outputs = self._model(input_ids=ids_batch, attention_mask=mask_batch)
                    val_loss += criterion(outputs.logits, y_batch).item() * ids_batch.size(0)
            avg_val_loss = val_loss / len(val_dataset)

            print(f"  epoch {epoch + 1:>3}/{self.epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

            # -- early stopping --
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = copy.deepcopy(self._model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.patience is not None and epochs_without_improvement >= self.patience:
                print(f"  Early stopping: no improvement for {self.patience} epochs")
                break

        # Restore best weights
        if best_weights is not None:
            self._model.load_state_dict(best_weights)
            print(f"  Restored best weights (val_loss={best_val_loss:.4f})")

        # Post-hoc temperature calibration on the validation set
        self._calibrate_temperature(val_loader, len(val_dataset))

    def _calibrate_temperature(self, val_loader: DataLoader, n_val: int) -> None:
        """Learn a scalar temperature on the validation set (Guo et al. 2017).

        Optimises T to minimise NLL on held-out logits. The learned value is
        stored in ``self._temperature`` and applied in ``predict()``.
        """
        self._model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for ids_batch, mask_batch, y_batch in val_loader:
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                outputs = self._model(input_ids=ids_batch, attention_mask=mask_batch)
                all_logits.append(outputs.logits.cpu())
                all_labels.append(y_batch)

        logits = torch.cat(all_logits)   # (n_val, 1)
        labels = torch.cat(all_labels)   # (n_val, 1)

        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
        criterion = nn.BCEWithLogitsLoss()

        def _eval():
            optimizer.zero_grad()
            loss = criterion(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        self._temperature = temperature.item()
        print(f"  Temperature calibration: T={self._temperature:.4f}")

    def predict(self, X_test: pd.DataFrame, return_proba: bool = False) -> pd.Series:
        """Tokenize test data and classify with the fine-tuned model.

        If *return_proba* is True, return raw probabilities instead of
        binary 0/1 predictions.
        """
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
                probs = torch.sigmoid(outputs.logits / self._temperature).squeeze(1)
                if return_proba:
                    preds_list.append(probs.cpu())
                else:
                    preds_list.append((probs >= 0.5).int().cpu())

        preds = torch.cat(preds_list).numpy()
        return pd.Series(preds, index=X_test.index)

    # -- save / load --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the trained model weights and hyperparameters to *path*.

        Creates a directory containing ``model.pt`` and ``meta.json``.
        Returns the directory path.
        """
        if self._model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._model.state_dict(), path / "model.pt")

        meta = {
            "hf_model_name": self.hf_model_name,
            "max_length": self.max_length,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "max_samples": self.max_samples,
            "freeze_backbone": self.freeze_backbone,
            "extraction_mode": self.extraction_mode,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "patience": self.patience,
            "focal_loss_gamma": self.focal_loss_gamma,
            "load_path": self.load_path,
            "temperature": self._temperature,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TransformerClassifier":
        """Load a previously saved model from *path*.

        Rebuilds the architecture from ``meta.json`` and loads weights
        from ``model.pt``.
        """
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)

        instance = cls(
            hf_model_name=meta["hf_model_name"],
            max_length=meta["max_length"],
            epochs=meta["epochs"],
            lr=meta["lr"],
            batch_size=meta["batch_size"],
            dropout=meta["dropout"],
            max_samples=meta["max_samples"],
            freeze_backbone=meta["freeze_backbone"],
            extraction_mode=meta.get("extraction_mode", "sentence"),
            focal_loss_gamma=meta.get("focal_loss_gamma", 0.0),
        )
        instance._load_model()
        instance._model.load_state_dict(torch.load(path / "model.pt", map_location=instance.device))
        instance._model.eval()
        instance._temperature = meta.get("temperature", 1.0)

        print(f"Model loaded from {path} (T={instance._temperature:.4f})")
        return instance
