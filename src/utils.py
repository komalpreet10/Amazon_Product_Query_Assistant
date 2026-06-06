"""
utils.py
--------
Shared utilities for corpus construction and tokenization.
Used by both bm25.py and semantic.py.

Usage:
    from src.utils import tokenize, build_corpus
"""

import re
import string
import logging
import json
from pathlib import Path

# ── logging ────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# Keep tokenization deterministic and free of import-time network downloads.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to",
    "was", "were", "will", "with",
}


# ── tokenizer ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenizes a string into a list of clean tokens.

    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Whitespace tokenize
        4. Remove stopwords
        5. Remove empty tokens

    Args:
        text: Raw input string.

    Returns:
        List of clean tokens.

    Example:
        >>> tokenize("Best moisturizer for sensitive skin!")
        ['best', 'moisturizer', 'sensitive', 'skin']
    """
    if not text or not isinstance(text, str):
        return []

    # lowercase
    text = text.lower()

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # whitespace tokenize
    tokens = text.split()

    # remove stopwords and empty tokens
    tokens = [t for t in tokens if t and t not in STOPWORDS]

    return tokens


# ── corpus builder ─────────────────────────────────────────────────────────

def build_corpus(products: list[dict]) -> tuple[list[str], list[list[str]]]:
    """
    Builds two parallel lists from products:
        - corpus: raw search_text strings (used by semantic search)
        - tokenized_corpus: tokenized search_text (used by BM25)

    Args:
        products: List of product dicts with 'search_text' field.

    Returns:
        Tuple of (corpus, tokenized_corpus)
            corpus            → list of raw search_text strings
            tokenized_corpus  → list of token lists
    """
    log.info("Building corpus from %s products...", f"{len(products):,}")

    corpus           = []
    tokenized_corpus = []

    for p in products:
        text = p.get("search_text", "").strip()
        if not text:
            continue
        corpus.append(text)
        tokenized_corpus.append(tokenize(text))

    log.info("Corpus built: %s documents", f"{len(corpus):,}")
    return corpus, tokenized_corpus


# ── load products ──────────────────────────────────────────────────────────

def load_products(path: Path = Path("data/processed/products.jsonl")) -> list[dict]:
    """
    Loads processed products from disk.

    Args:
        path: Path to products.jsonl file.

    Returns:
        List of product dicts.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Processed product catalog not found at {path}. "
            "Build or mount data/processed/products.jsonl before starting the service."
        )

    log.info("Loading products from %s...", path)
    with open(path, "r", encoding="utf-8") as f:
        products = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %s products", f"{len(products):,}")
    return products
