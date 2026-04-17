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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .base_kpub_classifier import KPUBClassifier, ensure_model
from data.compose import COMPOSE_FN
from .heads import HEADS

DEFAULT_HF_MODEL = "adsabs/astroBERT"

POOLING_MODES = ("cls", "mean")



def _pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce (batch, seq, hidden) to (batch, hidden)."""
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    # cls
    return hidden_states[:, 0, :]


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
        table: str = "keck",
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.0,
        patience: int | None = None,
        head: str = "mlp",
        pooling: str = "cls",
        load_path: str | None = None,
        device: str | None = None,
    ):
        if head not in HEADS:
            raise ValueError(f"Unknown head '{head}'. Choose from: {', '.join(HEADS)}")
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling '{pooling}'. Choose from: {', '.join(POOLING_MODES)}")
        self.head = head
        self.pooling = pooling
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
        self.load_path = load_path
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self._tokenizer = None
        self._backbone = None
        self._head = None
        self._temperature = 1.0  # learned via post-hoc calibration

    # -- internal helpers ---------------------------------------------------

    def _load_model(self):
        """Load tokenizer, backbone, and classification head."""
        model_path = ensure_model(self.hf_model_name)

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self._backbone is None:
            self._backbone = AutoModel.from_pretrained(model_path)
            hidden_dim = self._backbone.config.hidden_size
            head_cls = HEADS[self.head]
            self._head = head_cls(input_dim=hidden_dim, dropout=self.dropout)

            if self.load_path is not None:
                state_dict = torch.load(
                    Path(self.load_path) / "model.pt", map_location=self.device,
                )
                self._backbone.load_state_dict(state_dict["backbone"])
                self._head.load_state_dict(state_dict["head"])
                print(f"  Warm-start: loaded weights from {self.load_path}")

            if self.freeze_backbone:
                self._backbone.requires_grad_(False)

            self._backbone = self._backbone.to(self.device)
            self._head = self._head.to(self.device)

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run backbone → pool → head. Returns (batch, 1) logits."""
        hidden_states = self._backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = _pool(hidden_states, attention_mask, self.pooling)
        return self._head(pooled)

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

    def _parameters(self):
        """Return all trainable parameters across backbone and head."""
        for p in self._backbone.parameters():
            if p.requires_grad:
                yield p
        for p in self._head.parameters():
            if p.requires_grad:
                yield p

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

        use_val = self.patience is not None
        if use_val:
            n = len(input_ids)
            n_val = max(1, int(n * 0.1))
            indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            val_idx, train_idx = indices[:n_val], indices[n_val:]
            train_dataset = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
            val_dataset = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            train_dataset = TensorDataset(input_ids, attention_mask, labels)

        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self._backbone.train()
        self._head.train()

        optimizer = torch.optim.AdamW(
            self._parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        num_steps = len(loader) * self.epochs
        num_warmup = int(num_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_steps,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            # -- training --
            self._backbone.train()
            self._head.train()
            epoch_loss = 0.0
            for ids_batch, mask_batch, y_batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self._forward(ids_batch, mask_batch)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * ids_batch.size(0)

            avg_train_loss = epoch_loss / len(train_dataset)

            if not use_val:
                print(f"  epoch {epoch + 1:>3}/{self.epochs}  train_loss={avg_train_loss:.4f}")
                continue

            # -- validation --
            self._backbone.eval()
            self._head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for ids_batch, mask_batch, y_batch in val_loader:
                    ids_batch = ids_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    logits = self._forward(ids_batch, mask_batch)
                    val_loss += criterion(logits, y_batch).item() * ids_batch.size(0)
            avg_val_loss = val_loss / len(val_dataset)

            print(f"  epoch {epoch + 1:>3}/{self.epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

            # -- early stopping --
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {
                    "backbone": copy.deepcopy(self._backbone.state_dict()),
                    "head": copy.deepcopy(self._head.state_dict()),
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print(f"  Early stopping: no improvement for {self.patience} epochs")
                break

        # Restore best weights and calibrate temperature
        if best_weights is not None:
            self._backbone.load_state_dict(best_weights["backbone"])
            self._head.load_state_dict(best_weights["head"])
            print(f"  Restored best weights (val_loss={best_val_loss:.4f})")
            self._calibrate_temperature(val_loader)

    def _calibrate_temperature(self, val_loader: DataLoader) -> None:
        """Learn a scalar temperature on the validation set (Guo et al. 2017).

        Optimises T to minimise NLL on held-out logits. The learned value is
        stored in ``self._temperature`` and applied in ``predict()``.
        """
        self._backbone.eval()
        self._head.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for ids_batch, mask_batch, y_batch in val_loader:
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                logits = self._forward(ids_batch, mask_batch)
                all_logits.append(logits.cpu())
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
        if self._backbone is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        encoded = self._tokenize(X_test)
        dataset = TensorDataset(encoded["input_ids"], encoded["attention_mask"])
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self._backbone.eval()
        self._head.eval()
        preds_list = []

        with torch.no_grad():
            for ids_batch, mask_batch in loader:
                ids_batch = ids_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                logits = self._forward(ids_batch, mask_batch)
                probs = torch.sigmoid(logits / self._temperature).squeeze(1)
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
        if self._backbone is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            "backbone": self._backbone.state_dict(),
            "head": self._head.state_dict(),
        }, path / "model.pt")

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
            "head": self.head,
            "pooling": self.pooling,
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
            head=meta.get("head", "mlp"),
            pooling=meta.get("pooling", "cls"),
        )
        instance._load_model()

        state_dict = torch.load(path / "model.pt", map_location=instance.device)
        instance._backbone.load_state_dict(state_dict["backbone"])
        instance._head.load_state_dict(state_dict["head"])

        instance._backbone.eval()
        instance._head.eval()
        instance._temperature = meta.get("temperature", 1.0)

        print(f"Model loaded from {path} (T={instance._temperature:.4f})")
        return instance
