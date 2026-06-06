"""
bm25.py
-------
BM25 keyword-based retrieval system.
Builds, persists, and queries a BM25 index over the product corpus.

Usage:
    from src.bm25 import build_bm25, search_bm25, load_bm25

    # build once
    bm25 = build_bm25(tokenized_corpus, products)

    # load later
    bm25, tokenized_corpus = load_bm25()

    # search
    results = search_bm25(bm25, products, "moisturizer for sensitive skin", top_k=5)
"""

import pickle
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi
from src.utils import tokenize

# ── logging ────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
BM25_INDEX_PATH  = Path("data/processed/bm25_index.pkl")
CORPUS_PATH      = Path("data/processed/tokenized_corpus.pkl")


# ── build ──────────────────────────────────────────────────────────────────

def build_bm25(
    tokenized_corpus: list[list[str]],
    save_path: Path = BM25_INDEX_PATH,
    corpus_path: Path = CORPUS_PATH,
) -> BM25Okapi:
    """
    Builds a BM25 index from a tokenized corpus and saves it to disk.

    Args:
        tokenized_corpus: List of token lists (one per product).
        save_path:        Where to save the BM25 index.
        corpus_path:      Where to save the tokenized corpus.

    Returns:
        BM25Okapi index.
    """
    log.info("Building BM25 index over %s documents...", f"{len(tokenized_corpus):,}")

    bm25 = BM25Okapi(tokenized_corpus)

    # save index
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 index saved → %s", save_path)

    # save tokenized corpus
    with open(corpus_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    log.info("Tokenized corpus saved → %s", corpus_path)

    return bm25


# ── load ───────────────────────────────────────────────────────────────────

def load_bm25(
    save_path: Path  = BM25_INDEX_PATH,
    corpus_path: Path = CORPUS_PATH,
) -> tuple[BM25Okapi, list[list[str]]]:
    """
    Loads a persisted BM25 index and tokenized corpus from disk.

    Args:
        save_path:   Path to saved BM25 index.
        corpus_path: Path to saved tokenized corpus.

    Returns:
        Tuple of (BM25Okapi index, tokenized_corpus)
    """
    missing = [path for path in (save_path, corpus_path) if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"BM25 artifacts not found: {missing_list}. "
            "Build or mount data/processed artifacts before starting the service."
        )

    log.info("Loading BM25 index from %s...", save_path)
    with open(save_path, "rb") as f:
        bm25 = pickle.load(f)

    with open(corpus_path, "rb") as f:
        tokenized_corpus = pickle.load(f)

    log.info("BM25 index loaded.")
    return bm25, tokenized_corpus


# ── search ─────────────────────────────────────────────────────────────────

def search_bm25(
    bm25: BM25Okapi,
    products: list[dict],
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Searches the BM25 index for a query and returns top K products.

    Args:
        bm25:     BM25Okapi index.
        products: List of product dicts (same order as tokenized corpus).
        query:    Natural language query string.
        top_k:    Number of results to return.

    Returns:
        List of top K product dicts with an added 'bm25_score' field.
    """
    # tokenize query the same way as corpus
    query_tokens = tokenize(query)
    if not query_tokens:
        log.warning("Query tokenized to empty list: %s", query)
        return []

    # get scores for all documents
    scores = bm25.get_scores(query_tokens)

    # get top K indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # build results
    results = []
    for idx in top_indices:
        product = products[idx].copy()
        product["bm25_score"] = round(float(scores[idx]), 4)
        results.append(product)

    return results
