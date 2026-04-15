"""Compose and extract text features for publication classification.

Shared by all model types (transformer, embedding, etc.). Handles text
extraction (sentence, window, BM25, embedding modes) and per-table
composition of publication fields into classifier input.
"""

import re

from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from models.base_kpub_classifier import ensure_model

DEFAULT_HF_MODEL = "allenai/specter"

# ---------------------------------------------------------------------------
# Cached encoder for embedding-based extraction
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
# Text helpers
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

# ---------------------------------------------------------------------------
# Per-table composition functions
# ---------------------------------------------------------------------------

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

    archive_terms = ["koa", "archive", "archival", "download"]

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

def compose_drp_text(row: pd.Series, extraction_mode: str = "sentence") -> str:
    """Compose text for DRP (Data Reduction Pipeline) classification.

    Fields ordered by importance (text will be truncated).
    """
    full     = _safe(row.get("full"))
    abstract = _safe(row.get("abstract"))
    aff      = _safe(row.get("aff"))
    title    = _safe(row.get("title"))

    drp_terms = [
        "drp", "data reduction pipeline",
        "pypeit", "reduce",
    ] # data pipeline, data reduction

    parts = []
    if full:
        full = _remove_table_blocks(full)
        drp_sentences = _extract_relevant_sentences(full, terms=drp_terms, mode=extraction_mode)
        parts.append(f"FULL: {drp_sentences}")
    if abstract:
        parts.append(f"ABSTRACT: {abstract}")
    if aff:
        parts.append(f"AFFILIATIONS: {aff}")
    if title:
        parts.append(f"TITLE: {title}")

    return " ".join(parts)


COMPOSE_FN = {
    "keck": compose_keck_text,
    "small": compose_keck_text,
    "koa": compose_koa_text,
    "combined": compose_koa_text,
    "autokpub": compose_keck_text,
    "drp": compose_drp_text,
}
