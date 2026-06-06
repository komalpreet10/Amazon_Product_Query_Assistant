import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.reranker import rerank_products


class FakeCrossEncoder:
    def predict(self, pairs):
        return [0.1, 0.9, 0.3]


def test_rerank_products_orders_by_cross_encoder_score():
    products = [
        {"parent_asin": "A1", "title": "Basic cleanser", "hybrid_rank": 1},
        {"parent_asin": "A2", "title": "Sensitive skin moisturizer", "hybrid_rank": 2},
        {"parent_asin": "A3", "title": "Vitamin C serum", "hybrid_rank": 3},
    ]

    results = rerank_products(
        "moisturizer for sensitive skin",
        products,
        top_k=2,
        model=FakeCrossEncoder(),
    )

    assert [product["parent_asin"] for product in results] == ["A2", "A3"]
    assert results[0]["rerank_score"] == 0.9
    assert results[0]["original_hybrid_rank"] == 2
    assert results[0]["rerank_position"] == 1
    assert results[1]["original_hybrid_rank"] == 3
    assert results[1]["rerank_position"] == 2


def test_rerank_products_handles_empty_candidates():
    assert rerank_products("moisturizer", [], model=FakeCrossEncoder()) == []
