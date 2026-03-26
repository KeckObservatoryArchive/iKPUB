"""
Train a publication classifier using an embedding model with a PyTorch
classification head.

Embeddings are built at train/predict time from raw publication features
(not from pre-computed DB embeddings).
"""

from pathlib import Path

import re

from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base_kpub_classifier import KPUBClassifier, ensure_model

DEFAULT_HF_MODEL = "allenai/specter"  # alternatives: all-mpnet-base-v2

# ---------------------------------------------------------------------------
# Cached encoder for sentence extraction
# ---------------------------------------------------------------------------

_extraction_encoder: SentenceTransformer | None = None

def _get_extraction_encoder(model_name: str = DEFAULT_HF_MODEL) -> SentenceTransformer:
    """Return a cached SentenceTransformer for embedding-based extraction."""
    global _extraction_encoder
    if _extraction_encoder is None:
        model_path = ensure_model(model_name)
        _extraction_encoder = SentenceTransformer(str(model_path))
    return _extraction_encoder

# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class _ClassificationHead(nn.Module):
    """Small MLP: 768 → 256 → 64 → 1."""

    def __init__(self, input_dim: int = 768, dropout: float = 0.3):
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
# Text composition
# ---------------------------------------------------------------------------

def _safe(value) -> str:
    """Return value as a stripped string, or empty string for None/NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()

def _remove_table_blocks(text: str, min_prose_length: int = 60) -> str:
    """Remove runs of lines that look like table rows.

    A 'table block' is 3+ consecutive lines that are short and number-dense.
    """
    lines = text.splitlines()
    cleaned = []
    table_run = 0

    for line in lines:
        stripped = line.strip()
        # Lines with 3+ ellipsis markers are flattened table rows
        if stripped.count('· · ·') >= 3:
            continue
        is_short = len(stripped) < min_prose_length
        number_density = len(re.findall(r'\d', stripped)) / max(len(stripped), 1)
        has_ellipsis = '· · ·' in stripped or '...' in stripped
        looks_like_table = has_ellipsis or (is_short and number_density > 0.1)

        if looks_like_table:
            table_run += 1
        else:
            table_run = 0

        if table_run < 3:
            cleaned.append(line)

    return "\n".join(cleaned)


def _extract_relevant_sentences(text: str, terms: list[str] = ["keck"],
                                mode: str = "sentence", window: int = 24,
                                top_k: int = 6) -> str:
    """Extract text around occurrences of the given terms.

    Parameters
    ----------
    text : str
        Source text to search.
    terms : list[str]
        Keywords to match (case-insensitive).
    mode : str
        "sentence"  — return whole sentences containing a match.
        "window"    — return a fixed token window around each match.
        "bm25"      — return the top-k sentences ranked by BM25 relevance.
        "embedding" — return the top-k sentences ranked by embedding similarity.
    window : int
        Number of tokens to include on each side of the keyword (only used
        when mode="window").
    top_k : int
        Number of top sentences to return (used when mode="bm25" or "embedding").
    """
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)

    if mode == "sentence":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(s for s in sentences if pattern.search(s))

    if mode == "window":
        tokens = text.split()
        seen: set[int] = set()
        snippets: list[str] = []
        for i, tok in enumerate(tokens):
            if pattern.search(tok):
                start = max(0, i - window)
                end = min(len(tokens), i + window + 1)
                span = range(start, end)
                if not seen.intersection(span):
                    snippets.append(" ".join(tokens[start:end]))
                seen.update(span)
        return " ".join(snippets)

    if mode == "bm25":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return ""
        tokenized = [re.findall(r'\w+', s.lower()) for s in sentences]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores([t.lower() for t in terms])
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )
        return " ".join(sentences[i] for i in top_indices if scores[i] > 0)

    if mode == "embedding":
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return ""
        encoder = _get_extraction_encoder()
        query = " ".join(terms)
        embeddings = encoder.encode([query] + sentences, convert_to_numpy=True)
        query_emb = embeddings[0]
        sent_embs = embeddings[1:]
        scores = np.dot(sent_embs, query_emb) / (
            np.linalg.norm(sent_embs, axis=1) * np.linalg.norm(query_emb) + 1e-10
        )
        top_indices = sorted(
            sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        )
        return " ".join(sentences[i] for i in top_indices if scores[i] > 0)

    raise ValueError(f"Unknown extraction mode: {mode!r}")

def compose_keck_text(row: pd.Series, extraction_mode: str = "sentence") -> str:
    """Compose text for Keck publication classification.

    Fields ordered by importance (text will be truncated).
    """
    facility = _safe(row.get("facility"))
    full     = _safe(row.get("full"))
    abstract = _safe(row.get("abstract"))
    aff      = _safe(row.get("aff"))
    title    = _safe(row.get("title"))

    parts = []
    if facility:
        parts.append(f"FACILITY: {facility}")
    if full:
        keck_sentences = _extract_relevant_sentences(full, mode=extraction_mode)
        parts.append(f"FULL: {keck_sentences}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    if aff:
        parts.append(f"AFFILIATIONS: {aff}")
    if title:
        parts.append(f"TITLE: {title}")

    return " ".join(parts)


def compose_koa_text(row: pd.Series, extraction_mode: str = "sentence") -> str:
    """Compose text for KOA (Keck Observatory Archive) classification.

    Fields ordered by importance (text will be truncated).
    """
    full     = _safe(row.get("full"))
    abstract = _safe(row.get("abstract"))
    aff      = _safe(row.get("aff"))
    title    = _safe(row.get("title"))

    # archive_terms = ["koa", "archive"]
    archive_terms = ["koa", "archive", "pypeit", "nirc2", "lris", "hires", "nirspec", "smooth", "archived", "archival", "download", "repository", "dataset"] 
    
    parts = []
    if full:
        full = _remove_table_blocks(full)
        archive_sentences = _extract_relevant_sentences(full, terms=archive_terms, mode=extraction_mode)
        parts.append(f"FULL: {archive_sentences}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    if aff:
        parts.append(f"AFFILIATIONS: {aff}")
    if title:
        parts.append(f"TITLE: {title}")

    return " ".join(parts)

COMPOSE_FN = {
    "publications": compose_keck_text,
    "koa": compose_koa_text,
    "combined": compose_koa_text,
}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

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
        table: str = "publications",
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
        self._head: _ClassificationHead | None = None

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

        self._head = _ClassificationHead(input_dim=embedding_dim, dropout=self.dropout)
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
