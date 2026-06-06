from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import app, get_service
from app.schemas import AskRequest, SearchRequest


class FakeService:
    def health(self):
        return {
            "status": "ok",
            "products_loaded": 2,
            "bm25_loaded": True,
            "faiss_loaded": True,
            "embedding_model_loaded": True,
            "reranker_loaded": True,
        }

    def get_metrics(self):
        return {
            "searches_total": 1,
            "asks_total": 1,
            "errors_total": 0,
            "avg_retrieval_latency_ms": 12.5,
            "avg_reranking_latency_ms": 7.5,
            "avg_generation_latency_ms": 40.0,
            "last_error": None,
            "products_loaded": 2,
        }

    def search(self, query, top_k=5, mode="rerank"):
        return {
            "query": query,
            "mode": mode,
            "results": [
                {
                    "parent_asin": "A1",
                    "title": "Sensitive Skin Moisturizer",
                    "hybrid_score": 0.03,
                    "hybrid_rank": 3,
                    "original_hybrid_rank": 3,
                    "rerank_score": 0.88,
                    "rerank_position": 1,
                    "snippet": "Gentle moisturizer for sensitive skin.",
                }
            ],
            "retrieval_latency_ms": 6.0,
            "reranking_latency_ms": 4.0,
            "latency_ms": 10.0,
        }

    def ask(self, query, top_k=5):
        return {
            "response": "Try Sensitive Skin Moisturizer [1].",
            "sources": [
                {
                    "parent_asin": "A1",
                    "title": "Sensitive Skin Moisturizer",
                    "hybrid_rank": 2,
                    "original_hybrid_rank": 2,
                    "rerank_score": 0.91,
                    "rerank_position": 1,
                    "snippet": "Gentle moisturizer for sensitive skin.",
                }
            ],
            "retrieved": [
                {
                    "parent_asin": "A1",
                    "title": "Sensitive Skin Moisturizer",
                    "hybrid_rank": 2,
                    "original_hybrid_rank": 2,
                    "rerank_score": 0.91,
                    "rerank_position": 1,
                    "snippet": "Gentle moisturizer for sensitive skin.",
                }
            ],
            "filters": {},
            "context": "Product 1: Sensitive Skin Moisturizer",
            "citations": [
                {
                    "id": 1,
                    "parent_asin": "A1",
                    "title": "Sensitive Skin Moisturizer",
                }
            ],
            "guardrail_decision": "allowed",
            "retrieval_latency_ms": 5.0,
            "reranking_latency_ms": 5.0,
            "generation_latency_ms": 40.0,
            "latency_ms": 50.0,
        }


def client():
    app.dependency_overrides[get_service] = lambda: FakeService()
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health_endpoint():
    response = client().get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_endpoint():
    response = client().get("/ready")

    assert response.status_code == 200
    assert response.json()["reranker_loaded"] is True


def test_metrics_endpoint():
    response = client().get("/metrics")

    assert response.status_code == 200
    assert response.json()["searches_total"] == 1
    assert response.json()["avg_generation_latency_ms"] == 40.0


def test_search_endpoint():
    response = client().post(
        "/search",
        json={"query": "best moisturizer for sensitive skin", "top_k": 1, "mode": "rerank"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "rerank"
    assert data["retrieval_latency_ms"] == 6.0
    assert data["reranking_latency_ms"] == 4.0
    assert data["results"][0]["original_hybrid_rank"] == 3
    assert data["results"][0]["rerank_position"] == 1


def test_ask_endpoint_returns_citations():
    response = client().post(
        "/ask",
        json={"query": "best moisturizer for sensitive skin", "top_k": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert "[1]" in data["response"]
    assert data["sources"][0]["snippet"] == "Gentle moisturizer for sensitive skin."
    assert data["citations"][0]["parent_asin"] == "A1"


def test_search_request_rejects_blank_query():
    try:
        SearchRequest(query="   ", top_k=5)
    except ValidationError as exc:
        assert "query cannot be blank" in str(exc)
    else:
        raise AssertionError("SearchRequest accepted a blank query")


def test_ask_endpoint_schema_validation():
    response = client().post("/ask", json={"query": "   ", "top_k": 1})

    assert response.status_code == 422


def test_ask_request_trims_query():
    request = AskRequest(query="  best moisturizer for sensitive skin  ", top_k=3)

    assert request.query == "best moisturizer for sensitive skin"
