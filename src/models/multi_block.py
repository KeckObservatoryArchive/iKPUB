"""
Multi-block transformer classifier.

Extracts multiple 512-token blocks from the full text using different term
groups, runs a shared transformer backbone on each block to get CLS embeddings,
then concatenates those embeddings and classifies with an MLP.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from .base_kpub_classifier import KPUBClassifier, ensure_model
from .embedding import _safe, _extract_relevant_sentences

DEFAULT_HF_MODEL = "adsabs/astroBERT"

# ---------------------------------------------------------------------------
# Block term groups per table
# ---------------------------------------------------------------------------

BLOCK_TERMS: dict[str, list[list[str]]] = {
    "koa": [
        ["nirc2", "lris", "hires", "nirspec"],
        ["koa", "archive", "archived", "archival"],
        ["retrieved", "download", "repository", "dataset"],
    ],
    "publications": [
        ["keck", "wmko"],
        ["nirc2", "lris", "hires", "nirspec", "deimos", "mosfire", "kcwi", "osiris"],
        ["mauna", "maunakea", "observatory"],
    ],
}
BLOCK_TERMS["combined"] = BLOCK_TERMS["koa"]

# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------

def compose_block(row: pd.Series, terms: list[str],
                  extraction_mode: str = "bm25") -> str:
    """Compose a single block: extracted sentences from full text + context.

    Extracts relevant sentences from the full text using the given terms,
    then appends title and abstract to fill remaining token budget.
    """
    full = _safe(row.get("full"))
    title = _safe(row.get("title"))
    abstract = _safe(row.get("abstract"))

    parts = []
    if full:
        extracted = _extract_relevant_sentences(full, terms=terms, mode=extraction_mode)
        if extracted:
            parts.append(extracted)
    if title:
        parts.append(f"TITLE: {title}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class _MultiBlockHead(nn.Module):
    """MLP that takes averaged CLS embeddings from all blocks."""

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MultiBlockClassifier(KPUBClassifier):
    """Multi-block transformer: multiple full-text extractions + shared backbone + MLP."""

    def __init__(
        self,
        hf_model_name: str = DEFAULT_HF_MODEL,
        max_length: int = 512,
        epochs: int = 10,
        lr: float = 2e-5,
        batch_size: int = 8,
        dropout: float = 0.3,
        max_samples: int | None = None,
        freeze_backbone: bool = False,
        extraction_mode: str = "bm25",
        table: str = "publications",
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
        patience: int | None = None,
        focal_loss_gamma: float = 0.0,
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
        self.device = device or ("mps" if torch.mps.is_available() else "cpu")

        self.block_terms = BLOCK_TERMS.get(table, BLOCK_TERMS["publications"])
        self.n_blocks = len(self.block_terms)

        self._tokenizer = None
        self._backbone = None
        self._head: _MultiBlockHead | None = None
        self._temperature = 1.0

    # -- internal helpers ---------------------------------------------------

    def _load_model(self):
        """Load tokenizer and backbone (no classification head — we add our own)."""
        model_path = ensure_model(self.hf_model_name)

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self._backbone is None:
            self._backbone = AutoModel.from_pretrained(model_path)
            if self.freeze_backbone:
                for param in self._backbone.parameters():
                    param.requires_grad = False
            self._backbone.to(self.device)

        if self._head is None:
            hidden_dim = self._backbone.config.hidden_size
            self._head = _MultiBlockHead(
                input_dim=hidden_dim, dropout=self.dropout,
            )
            self._head.to(self.device)

    def _tokenize_texts(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize a list of strings in batches."""
        all_input_ids = []
        all_attention_masks = []
        tok_batch = 64
        for i in range(0, len(texts), tok_batch):
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

    def _compose_all_blocks(self, X: pd.DataFrame) -> list[list[str]]:
        """Compose text for each block for each example.

        Returns a list of n_blocks lists, each with len(X) strings.
        """
        blocks = [[] for _ in range(self.n_blocks)]
        for _, row in tqdm(X.iterrows(), total=len(X), desc="Composing blocks"):
            for b, terms in enumerate(self.block_terms):
                text = compose_block(row, terms, extraction_mode=self.extraction_mode)
                blocks[b].append(text)
        return blocks

    def _tokenize_all_blocks(self, X: pd.DataFrame) -> list[dict[str, torch.Tensor]]:
        """Compose and tokenize all blocks. Returns a list of n_blocks token dicts."""
        blocks_text = self._compose_all_blocks(X)
        tokenized = []
        for b in range(self.n_blocks):
            print(f"  Tokenizing block {b + 1}/{self.n_blocks} (terms: {self.block_terms[b]})")
            tokenized.append(self._tokenize_texts(blocks_text[b]))
        return tokenized

    def _forward_blocks(self, block_ids: list[torch.Tensor],
                        block_masks: list[torch.Tensor]) -> torch.Tensor:
        """Run backbone on each block's tokens and return averaged CLS embeddings.

        This keeps everything in the computation graph so gradients flow
        back through the backbone.
        """
        cls_list = []
        for ids, mask in zip(block_ids, block_masks):
            outputs = self._backbone(input_ids=ids, attention_mask=mask)
            cls_list.append(outputs.last_hidden_state[:, 0, :])
        return torch.stack(cls_list).mean(dim=0)

    def _encode_blocks_no_grad(self, X: pd.DataFrame) -> torch.Tensor:
        """Encode all blocks without gradients (for inference/calibration)."""
        tokenized = self._tokenize_all_blocks(X)

        all_cls = []
        self._backbone.eval()
        for b in range(self.n_blocks):
            input_ids = tokenized[b]["input_ids"]
            attention_mask = tokenized[b]["attention_mask"]
            cls_embeddings = []
            dataset = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset, batch_size=self.batch_size)
            with torch.no_grad():
                for ids_batch, mask_batch in loader:
                    ids_batch = ids_batch.to(self.device)
                    mask_batch = mask_batch.to(self.device)
                    outputs = self._backbone(input_ids=ids_batch, attention_mask=mask_batch)
                    cls_embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
            all_cls.append(torch.cat(cls_embeddings))

        return torch.stack(all_cls).mean(dim=0)

    # -- public API ---------------------------------------------------------

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._load_model()

        if self.max_samples is not None and len(X_train) > self.max_samples:
            X_train = X_train.sample(n=self.max_samples, random_state=42)
            y_train = y_train.loc[X_train.index]
            print(f"  Subsampled to {self.max_samples} training examples")

        # Tokenize all blocks upfront (text composition is expensive, do it once)
        print("Tokenizing training blocks...")
        tokenized_blocks = self._tokenize_all_blocks(X_train)
        labels = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        # Train/val split
        n = labels.size(0)
        n_val = max(1, int(n * 0.1))
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        # Build datasets: each example has (block0_ids, block0_mask, block1_ids, ..., label)
        train_tensors = []
        val_tensors = []
        for b in range(self.n_blocks):
            train_tensors.extend([
                tokenized_blocks[b]["input_ids"][train_idx],
                tokenized_blocks[b]["attention_mask"][train_idx],
            ])
            val_tensors.extend([
                tokenized_blocks[b]["input_ids"][val_idx],
                tokenized_blocks[b]["attention_mask"][val_idx],
            ])
        train_tensors.append(labels[train_idx])
        val_tensors.append(labels[val_idx])

        train_dataset = TensorDataset(*train_tensors)
        val_dataset = TensorDataset(*val_tensors)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        if self.focal_loss_gamma > 0:
            from .transformer import _focal_bce_with_logits
            def criterion(logits, targets):
                return _focal_bce_with_logits(logits, targets, self.focal_loss_gamma)
            print(f"  Using focal loss (gamma={self.focal_loss_gamma})")
        else:
            criterion = nn.BCEWithLogitsLoss()

        if self.freeze_backbone:
            params = self._head.parameters()
        else:
            params = list(self._backbone.parameters()) + list(self._head.parameters())

        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        num_steps = len(loader) * self.epochs
        num_warmup = int(num_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup, num_training_steps=num_steps,
        )

        best_val_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self._backbone.train()
            self._head.train()
            epoch_loss = 0.0

            for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                # Unpack: pairs of (ids, mask) per block, then label
                block_ids = []
                block_masks = []
                for b in range(self.n_blocks):
                    block_ids.append(batch[b * 2].to(self.device))
                    block_masks.append(batch[b * 2 + 1].to(self.device))
                y_batch = batch[-1].to(self.device)

                combined_cls = self._forward_blocks(block_ids, block_masks)
                logits = self._head(combined_cls)
                loss = criterion(logits, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item() * y_batch.size(0)

            avg_train_loss = epoch_loss / len(train_dataset)

            # Validation
            self._backbone.eval()
            self._head.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    block_ids = []
                    block_masks = []
                    for b in range(self.n_blocks):
                        block_ids.append(batch[b * 2].to(self.device))
                        block_masks.append(batch[b * 2 + 1].to(self.device))
                    y_batch = batch[-1].to(self.device)

                    combined_cls = self._forward_blocks(block_ids, block_masks)
                    logits = self._head(combined_cls)
                    val_loss += criterion(logits, y_batch).item() * y_batch.size(0)
            avg_val_loss = val_loss / len(val_dataset)

            print(f"  epoch {epoch + 1:>3}/{self.epochs}  train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = {
                    "backbone": copy.deepcopy(self._backbone.state_dict()),
                    "head": copy.deepcopy(self._head.state_dict()),
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.patience is not None and epochs_without_improvement >= self.patience:
                print(f"  Early stopping: no improvement for {self.patience} epochs")
                break

        if best_weights is not None:
            self._backbone.load_state_dict(best_weights["backbone"])
            self._head.load_state_dict(best_weights["head"])
            print(f"  Restored best weights (val_loss={best_val_loss:.4f})")

        # Temperature calibration on val set
        self._calibrate_temperature(val_loader)

    def _calibrate_temperature(self, val_loader: DataLoader) -> None:
        self._backbone.eval()
        self._head.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                block_ids = []
                block_masks = []
                for b in range(self.n_blocks):
                    block_ids.append(batch[b * 2].to(self.device))
                    block_masks.append(batch[b * 2 + 1].to(self.device))
                y_batch = batch[-1]

                combined_cls = self._forward_blocks(block_ids, block_masks)
                logits = self._head(combined_cls)
                all_logits.append(logits.cpu())
                all_labels.append(y_batch)

        logits = torch.cat(all_logits)
        labels = torch.cat(all_labels)

        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)
        crit = nn.BCEWithLogitsLoss()

        def _eval():
            optimizer.zero_grad()
            loss = crit(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        self._temperature = temperature.item()
        print(f"  Temperature calibration: T={self._temperature:.4f}")

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        if self._head is None:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        print("Encoding test blocks...")
        X_emb = self._encode_blocks_no_grad(X_test)

        self._head.eval()
        dataset = TensorDataset(X_emb)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        preds_list = []
        with torch.no_grad():
            for (emb_batch,) in loader:
                emb_batch = emb_batch.to(self.device)
                logits = self._head(emb_batch)
                probs = torch.sigmoid(logits / self._temperature).squeeze(1)
                preds_list.append((probs >= 0.5).int().cpu())

        preds = torch.cat(preds_list).numpy()
        return pd.Series(preds, index=X_test.index)

    # -- save / load --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        if self._backbone is None or self._head is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(self._backbone.state_dict(), path / "backbone.pt")
        torch.save(self._head.state_dict(), path / "head.pt")

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
            "table": self.table,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "patience": self.patience,
            "focal_loss_gamma": self.focal_loss_gamma,
            "temperature": self._temperature,
            "n_blocks": self.n_blocks,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MultiBlockClassifier":
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
            extraction_mode=meta.get("extraction_mode", "bm25"),
            table=meta.get("table", "publications"),
            focal_loss_gamma=meta.get("focal_loss_gamma", 0.0),
        )
        instance._load_model()
        instance._backbone.load_state_dict(
            torch.load(path / "backbone.pt", map_location=instance.device)
        )
        instance._head.load_state_dict(
            torch.load(path / "head.pt", map_location=instance.device)
        )
        instance._backbone.eval()
        instance._head.eval()
        instance._temperature = meta.get("temperature", 1.0)

        print(f"Model loaded from {path} (T={instance._temperature:.4f})")
        return instance
