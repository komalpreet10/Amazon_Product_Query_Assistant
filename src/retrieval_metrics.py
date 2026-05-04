"""
retrieval_metrics.py
--------------------
Evaluation metrics for retrieval systems.

Metrics:
    - Precision@K
    - MRR (Mean Reciprocal Rank)
    - NDCG@K (Normalized Discounted Cumulative Gain)

Usage:
    from src.retrieval_metrics import evaluate, evaluate_all

    # single query
    scores = evaluate(retrieved_ids, relevant_ids, k=5)

    # all queries
    results = evaluate_all(bm25, index, products, ground_truth, model, k=5)
"""

import math
import logging
from src.bm25 import search_bm25
from src.semantic import search_semantic
from src.hybrid import hybrid_search

log = logging.getLogger(__name__)


# ── core metrics ───────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K = number of relevant items in top K / K

    Args:
        retrieved: Ordered list of retrieved product IDs.
        relevant:  List of relevant product IDs (ground truth).
        k:         Cutoff rank.

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    retrieved_at_k = retrieved[:k]
    relevant_set   = set(relevant)
    hits           = sum(1 for r in retrieved_at_k if r in relevant_set)
    return hits / k


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """
    Mean Reciprocal Rank = 1 / rank of first relevant result.
    Returns 0 if no relevant result found.

    Args:
        retrieved: Ordered list of retrieved product IDs.
        relevant:  List of relevant product IDs (ground truth).

    Returns:
        MRR score (0.0 to 1.0)
    """
    relevant_set = set(relevant)
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    Rewards relevant results appearing higher in the ranked list.

    Args:
        retrieved: Ordered list of retrieved product IDs.
        relevant:  List of relevant product IDs (ground truth).
        k:         Cutoff rank.

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    relevant_set = set(relevant)

    # DCG — actual ranking
    dcg = 0.0
    for rank, r in enumerate(retrieved[:k], start=1):
        if r in relevant_set:
            dcg += 1 / math.log2(rank + 1)

    # IDCG — ideal ranking (all relevant items at top)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(retrieved: list[str], relevant: list[str], k: int = 5) -> dict:
    """
    Computes all metrics for a single query.

    Args:
        retrieved: Ordered list of retrieved product IDs.
        relevant:  List of relevant product IDs (ground truth).
        k:         Cutoff rank.

    Returns:
        Dict with precision@k, mrr, ndcg@k scores.
    """
    return {
        f"precision@{k}": round(precision_at_k(retrieved, relevant, k), 4),
        "mrr":             round(mrr(retrieved, relevant), 4),
        f"ndcg@{k}":       round(ndcg_at_k(retrieved, relevant, k), 4),
    }


# ── full evaluation ────────────────────────────────────────────────────────

def evaluate_all(
    bm25,
    index,
    products: list[dict],
    ground_truth: dict[str, list[str]],
    model,
    k: int = 5,
) -> dict:
    """
    Evaluates BM25, Semantic, and Hybrid across all queries.

    Args:
        bm25:         BM25Okapi index.
        index:        FAISS index.
        products:     List of product dicts.
        ground_truth: Dict of {query: [relevant_ids]}.
        model:        SentenceTransformer model.
        k:            Cutoff rank.

    Returns:
        Dict with per-query and average scores for each method.
    """
    results = {"bm25": {}, "semantic": {}, "hybrid": {}}

    for query, relevant in ground_truth.items():
        # get retrieved IDs for each method
        bm25_ids     = [r["parent_asin"] for r in search_bm25(bm25, products, query, top_k=k)]
        semantic_ids = [r["parent_asin"] for r in search_semantic(index, products, query, top_k=k, model=model)]
        hybrid_ids   = [r["parent_asin"] for r in hybrid_search(bm25, index, products, query, top_k=k, model=model)]

        results["bm25"][query]     = evaluate(bm25_ids, relevant, k)
        results["semantic"][query] = evaluate(semantic_ids, relevant, k)
        results["hybrid"][query]   = evaluate(hybrid_ids, relevant, k)

    # compute averages
    for method in ["bm25", "semantic", "hybrid"]:
        scores = results[method]
        avg = {}
        for metric in [f"precision@{k}", "mrr", f"ndcg@{k}"]:
            avg[metric] = round(sum(s[metric] for s in scores.values()) / len(scores), 4)
        results[method]["average"] = avg
        log.info("%s average: %s", method.upper(), avg)

    return results