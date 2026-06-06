"""
hybrid.py
---------
Hybrid retrieval combining BM25 and semantic search using
Reciprocal Rank Fusion (RRF).

Usage:
    from src.hybrid import hybrid_search

    results = hybrid_search(bm25, index, products, query, top_k=5, model=model)
"""

import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.bm25 import search_bm25
from src.semantic import search_semantic

log = logging.getLogger(__name__)

# RRF constant — reduces impact of very high rankings
RRF_K = 60


def hybrid_search(
    bm25: BM25Okapi,
    index,
    products: list[dict],
    query: str,
    top_k: int = 5,
    model: SentenceTransformer = None,
) -> list[dict]:
    """
    Combines BM25 and semantic search using Reciprocal Rank Fusion.

    Args:
        bm25:     BM25Okapi index.
        index:    FAISS index.
        products: List of product dicts.
        query:    Natural language query string.
        top_k:    Number of results to return.
        model:    SentenceTransformer model.

    Returns:
        List of top K product dicts with 'hybrid_score' field.
    """

    # get more candidates from each method for better fusion
    candidates = top_k * 2

    # get results from both methods
    bm25_results     = search_bm25(bm25, products, query, top_k=candidates)
    semantic_results = search_semantic(index, products, query, top_k=candidates, model=model)

    # build parent_asin → rank mapping for each method
    bm25_ranks     = {r["parent_asin"]: rank for rank, r in enumerate(bm25_results)}
    semantic_ranks = {r["parent_asin"]: rank for rank, r in enumerate(semantic_results)}

    # collect all unique products
    all_asins = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

    # compute RRF score for each product
    rrf_scores = {}
    for asin in all_asins:
        bm25_rank     = bm25_ranks.get(asin, candidates)      # penalize if not in results
        semantic_rank = semantic_ranks.get(asin, candidates)   # penalize if not in results
        rrf_scores[asin] = 1 / (bm25_rank + RRF_K) + 1 / (semantic_rank + RRF_K)

    # sort by RRF score
    ranked_asins = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    # build product lookup
    product_lookup = {p["parent_asin"]: p for p in products}

    # build final results
    results = []
    for hybrid_rank, asin in enumerate(ranked_asins, start=1):
        if asin not in product_lookup:
            continue
        product = product_lookup[asin].copy()
        product["hybrid_score"]   = round(rrf_scores[asin], 6)
        product["hybrid_rank"]    = hybrid_rank
        product["bm25_rank"]      = bm25_ranks.get(asin, None)
        product["semantic_rank"]  = semantic_ranks.get(asin, None)
        results.append(product)

    return results
