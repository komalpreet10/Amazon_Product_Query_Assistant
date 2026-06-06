from pathlib import Path

from src import retrieval_metrics


class FakeReranker:
    def predict(self, pairs):
        return [0.1, 0.9]


def test_evaluate_with_reranking_saves_comparison(monkeypatch, tmp_path):
    products = [
        {"parent_asin": "A1", "title": "Basic cleanser"},
        {"parent_asin": "A2", "title": "Sensitive skin moisturizer"},
    ]

    def fake_search(*args, **kwargs):
        return products

    monkeypatch.setattr(retrieval_metrics, "search_bm25", fake_search)
    monkeypatch.setattr(retrieval_metrics, "search_semantic", fake_search)
    monkeypatch.setattr(retrieval_metrics, "hybrid_search", fake_search)

    output_path = Path(tmp_path) / "evaluation_metrics.json"
    results = retrieval_metrics.evaluate_with_reranking(
        bm25=None,
        index=None,
        products=products,
        ground_truth={"best moisturizer for sensitive skin": ["A2"]},
        model=None,
        reranker_model=FakeReranker(),
        k=1,
        output_path=output_path,
    )

    assert output_path.exists()
    assert results["comparison"]["hybrid_average"]["precision@1"] == 0.0
    assert results["comparison"]["hybrid_reranked_average"]["precision@1"] == 1.0
