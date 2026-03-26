"""Discover which tokens/phrases drive predictions in a trained transformer.

Uses Integrated Gradients (via captum) on the embedding layer to compute
per-token attribution scores across labeled examples. Aggregates results
into ranked token/phrase lists to inform extraction term selection.

Usage:
    python src/eval/mine_salience.py --load data/models/trained/transformer_2026-... --table koa
    python src/eval/mine_salience.py --load <path> --table publications --top-k 100
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_pipes.prepare import load_publications
from models.transformer import TransformerClassifier, _get_backbone

PROJECT_ROOT = Path(__file__).parents[2]
DB_PATH = PROJECT_ROOT / "data" / "pubs" / "kpub.db"
OUTPUT_DIR = PROJECT_ROOT / "out" / "salience"


def _merge_subwords(tokens: list[str], scores: list[float]) -> list[tuple[str, float]]:
    """Merge subword tokens back into whole words with summed scores.

    Handles both tokenizer conventions:
    - BERT-style: continuation tokens start with '##', others are word-initial
    - RoBERTa-style: word-initial tokens start with 'Ġ'
    """
    # Detect style: if any token starts with ##, it's BERT-style
    is_bert = any(t.startswith("##") for t in tokens)

    words: list[tuple[str, float]] = []
    for tok, score in zip(tokens, scores):
        if is_bert:
            if tok.startswith("##"):
                if words:
                    prev_word, prev_score = words[-1]
                    words[-1] = (prev_word + tok[2:], prev_score + score)
                continue
            words.append((tok, score))
        else:
            if tok.startswith("Ġ") or not words:
                words.append((tok.replace("Ġ", ""), score))
            else:
                prev_word, prev_score = words[-1]
                words[-1] = (prev_word + tok, prev_score + score)
    return words


def _extract_ngrams(words: list[tuple[str, float]], n: int) -> list[tuple[str, float]]:
    """Extract n-grams from word list with summed attribution scores."""
    ngrams = []
    for i in range(len(words) - n + 1):
        phrase = " ".join(w for w, _ in words[i : i + n])
        score = sum(s for _, s in words[i : i + n])
        ngrams.append((phrase, score))
    return ngrams


def _attribute_chunk(
    lig: LayerIntegratedGradients,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: str,
    n_steps: int,
) -> tuple[list[str], list[float]]:
    """Run IG on a single chunk and return (tokens, scores) without special tokens."""
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id)

    attrs = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
    )

    token_attrs = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    seq_len = attention_mask.sum().item()
    token_ids = input_ids.squeeze(0)[:seq_len].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    scores = token_attrs[:seq_len].tolist()

    # Strip special tokens (<s>, </s>)
    return tokens[1:-1], scores[1:-1]


def compute_attributions(
    model: TransformerClassifier,
    texts: list[str],
    labels: np.ndarray,
    n_steps: int = 50,
    stride: int = 256,
) -> list[dict]:
    """Run Integrated Gradients on each example and return per-token attributions.

    For texts longer than the model's max_length, uses a sliding window with
    the given stride. Attribution scores are averaged where chunks overlap.
    """
    tokenizer = model._tokenizer
    hf_model = model._model
    hf_model.eval()
    device = model.device
    max_len = model.max_length

    backbone = _get_backbone(hf_model)
    embedding_layer = backbone.embeddings

    def forward_fn(input_ids, attention_mask):
        outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)

    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    # Content tokens per chunk (minus 2 for <s> and </s>)
    content_len = max_len - 2

    results = []
    for i, text in enumerate(tqdm(texts, desc="Computing attributions")):
        # Tokenize the full text without truncation
        full_encoded = tokenizer(text, add_special_tokens=False)
        all_token_ids = full_encoded["input_ids"]

        if len(all_token_ids) <= content_len:
            # Fits in one chunk — simple path
            encoded = tokenizer(
                text, padding="max_length", truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            tokens, scores = _attribute_chunk(
                lig, tokenizer, encoded["input_ids"], encoded["attention_mask"],
                device, n_steps,
            )
        else:
            # Sliding window over the full token sequence
            n_tokens = len(all_token_ids)
            score_sums = np.zeros(n_tokens)
            score_counts = np.zeros(n_tokens)

            bos_id = tokenizer.bos_token_id or tokenizer.cls_token_id
            eos_id = tokenizer.eos_token_id or tokenizer.sep_token_id

            for start in range(0, n_tokens, stride):
                end = min(start + content_len, n_tokens)
                chunk_ids = all_token_ids[start:end]

                # Wrap with special tokens and pad
                input_ids = [bos_id] + chunk_ids + [eos_id]
                attn_mask = [1] * len(input_ids)
                pad_len = max_len - len(input_ids)
                input_ids += [tokenizer.pad_token_id] * pad_len
                attn_mask += [0] * pad_len

                input_ids_t = torch.tensor([input_ids])
                attn_mask_t = torch.tensor([attn_mask])

                _, chunk_scores = _attribute_chunk(
                    lig, tokenizer, input_ids_t, attn_mask_t, device, n_steps,
                )

                # Map chunk scores back to global positions
                for j, score in enumerate(chunk_scores):
                    score_sums[start + j] += score
                    score_counts[start + j] += 1

                if end >= n_tokens:
                    break

            # Average overlapping scores
            avg_scores = np.divide(score_sums, score_counts, where=score_counts > 0)
            tokens = tokenizer.convert_ids_to_tokens(all_token_ids)
            scores = avg_scores.tolist()

        results.append({
            "index": i,
            "label": int(labels[i]),
            "tokens": tokens,
            "scores": scores,
        })

    return results


def aggregate_results(
    results: list[dict], top_k: int = 50,
) -> dict:
    """Aggregate per-example attributions into ranked token/phrase lists."""
    word_scores: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "count": 0})
    word_scores_by_class: dict[int, dict[str, dict]] = {
        0: defaultdict(lambda: {"total": 0.0, "count": 0}),
        1: defaultdict(lambda: {"total": 0.0, "count": 0}),
    }
    bigram_scores: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "count": 0})

    for r in results:
        words = _merge_subwords(r["tokens"], r["scores"])
        label = r["label"]

        for word, score in words:
            w = word.lower().strip()
            if len(w) < 2 or not re.search(r"[a-z]", w):
                continue
            word_scores[w]["total"] += score
            word_scores[w]["count"] += 1
            word_scores_by_class[label][w]["total"] += score
            word_scores_by_class[label][w]["count"] += 1

        for phrase, score in _extract_ngrams(words, 2):
            p = phrase.lower().strip()
            bigram_scores[p]["total"] += score
            bigram_scores[p]["count"] += 1

    def _rank(scores_dict, k):
        ranked = [
            {"token": tok, "mean_score": d["total"] / d["count"], "count": d["count"]}
            for tok, d in scores_dict.items()
            if d["count"] >= 3  # filter noise from rare tokens
        ]
        ranked.sort(key=lambda x: x["mean_score"], reverse=True)
        return ranked[:k]

    return {
        "top_tokens": _rank(word_scores, top_k),
        "top_tokens_positive": _rank(word_scores_by_class[1], top_k),
        "top_tokens_negative": _rank(word_scores_by_class[0], top_k),
        "top_bigrams": _rank(bigram_scores, top_k),
    }


def main():
    parser = argparse.ArgumentParser(description="Mine salient tokens from a trained transformer")
    parser.add_argument("--load", required=True, help="Path to saved TransformerClassifier")
    parser.add_argument("--table", default="publications", help="DB table (default: publications)")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top tokens to report")
    parser.add_argument("--holdout-table", metavar="TABLE",
                        help="Define test set from this table's bibcodes")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Integration steps for IG (higher = more precise, slower)")
    parser.add_argument("--n-papers", type=int, default=None,
                        help="Number of papers to analyze (default: all in test set)")
    args = parser.parse_args()

    # Load model
    model = TransformerClassifier.load(args.load)

    # Load data — use full text, not extracted
    pubs = load_publications(DB_PATH, query=f"SELECT * FROM {args.table} WHERE year < 2024 AND year > 1999")

    if args.holdout_table:
        holdout_pubs = load_publications(
            DB_PATH, query=f"SELECT bibcode, keck_manual FROM {args.holdout_table} WHERE year < 2024 AND year > 1999",
        )
        _, test_bibcodes = train_test_split(holdout_pubs["bibcode"], test_size=0.2, random_state=42)
        is_test = pubs["bibcode"].isin(set(test_bibcodes))
        pubs = pubs[is_test]
    else:
        _, pubs = train_test_split(pubs, test_size=0.2, random_state=42)

    if args.n_papers is not None:
        pubs = pubs.sample(n=min(args.n_papers, len(pubs)), random_state=42)

    labels = pubs["keck_manual"].values

    # Use raw full text for attribution (the whole point is to see what matters
    # without pre-filtering)
    texts = []
    for _, row in pubs.iterrows():
        full = row.get("full", "") or ""
        abstract = row.get("abstract", "") or ""
        title = row.get("title", "") or ""
        texts.append(f"{title} {abstract} {full}")

    # Run attributions
    results = compute_attributions(model, texts, labels, n_steps=args.n_steps)

    # Aggregate
    aggregated = aggregate_results(results, top_k=args.top_k)

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"salience_{args.table}.json"
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Top {args.top_k} salient tokens (all examples):")
    print(f"{'='*60}")
    for entry in aggregated["top_tokens"]:
        print(f"  {entry['token']:<30} mean={entry['mean_score']:.4f}  n={entry['count']}")

    print(f"\nTop {args.top_k} salient tokens (POSITIVE class):")
    print(f"{'-'*60}")
    for entry in aggregated["top_tokens_positive"]:
        print(f"  {entry['token']:<30} mean={entry['mean_score']:.4f}  n={entry['count']}")

    print(f"\nTop {args.top_k} salient bigrams:")
    print(f"{'-'*60}")
    for entry in aggregated["top_bigrams"]:
        print(f"  {entry['token']:<30} mean={entry['mean_score']:.4f}  n={entry['count']}")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
