"""
reranker.py
-----------
Cross-encoder reranking for hybrid retrieval candidates.
"""

import logging
from typing import Protocol

from sentence_transformers import CrossEncoder

log = logging.getLogger(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class PredictableModel(Protocol):
    def predict(self, sentences):
        ...


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """
    Loads the cross-encoder used to rerank RRF candidates.
    """
    log.info("Loading reranker model: %s", model_name)
    return CrossEncoder(model_name)


def _product_text(product: dict) -> str:
    """
    Builds a concise reranker document from an existing product record.
    """
    if product.get("search_text"):
        return product["search_text"]

    parts = [
        product.get("title", ""),
        product.get("store", ""),
        " ".join(product.get("features") or []),
        " ".join(product.get("description") or []),
    ]
    if product.get("top_reviews"):
        parts.extend(
            f"{review.get('title', '')}. {review.get('text', '')}"
            for review in product["top_reviews"][:2]
        )
    return " | ".join(part for part in parts if part)


def rerank_products(
    query: str,
    products: list[dict],
    top_k: int = 5,
    model: PredictableModel | None = None,
) -> list[dict]:
    """
    Reranks retrieved products with a cross-encoder.

    Args:
        query: User query.
        products: Candidate products, usually from hybrid RRF.
        top_k: Number of reranked products to return.
        model: Optional injected cross-encoder-compatible model.

    Returns:
        Products ordered by cross-encoder score, each with rerank_score and
        rerank_position fields.
    """
    if not products:
        return []

    if model is None:
        model = load_reranker()

    pairs = [(query, _product_text(product)) for product in products]
    scores = model.predict(pairs)

    scored = []
    for original_position, (product, score) in enumerate(zip(products, scores), start=1):
        item = product.copy()
        item["original_hybrid_rank"] = item.get("hybrid_rank", original_position)
        item["rerank_score"] = round(float(score), 6)
        scored.append(item)

    scored.sort(key=lambda product: product["rerank_score"], reverse=True)

    results = scored[:top_k]
    for position, product in enumerate(results, start=1):
        product["rerank_position"] = position

    log.info("Reranked %s candidates to top %s", len(products), len(results))
    return results
