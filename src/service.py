"""
service.py
----------
Application service layer for production search and RAG endpoints.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

from sentence_transformers import SentenceTransformer

from src.bm25 import load_bm25, search_bm25
from src.guardrails import check_input
from src.hybrid import hybrid_search
from src.rag import rag_answer
from src.reranker import load_reranker, rerank_products
from src.semantic import load_semantic_index, search_semantic
from src.utils import load_products

log = logging.getLogger(__name__)


def compact_product(product: dict) -> dict:
    """
    Keeps API payloads focused on product and ranking fields.
    """
    keys = [
        "parent_asin",
        "title",
        "store",
        "price",
        "average_rating",
        "rating_number",
        "brand",
        "item_form",
        "bm25_score",
        "semantic_score",
        "hybrid_score",
        "hybrid_rank",
        "original_hybrid_rank",
        "bm25_rank",
        "semantic_rank",
        "rerank_score",
        "rerank_position",
    ]
    item = {key: product.get(key) for key in keys if key in product}
    if product.get("top_reviews"):
        review = product["top_reviews"][0]
        text = review.get("text")
        item["top_review"] = {
            "title": review.get("title"),
            "text": text,
            "rating": review.get("rating"),
        }
        if text:
            item["snippet"] = text[:240]
    elif product.get("search_text"):
        item["snippet"] = product["search_text"][:240]
    return item


@dataclass
class ServiceMetrics:
    searches_total: int = 0
    asks_total: int = 0
    errors_total: int = 0
    retrieval_latency_ms_total: float = 0.0
    reranking_latency_ms_total: float = 0.0
    generation_latency_ms_total: float = 0.0
    last_error: str | None = None
    lock: Lock = field(default_factory=Lock)

    def record_search(self, retrieval_ms: float, reranking_ms: float = 0.0) -> None:
        with self.lock:
            self.searches_total += 1
            self.retrieval_latency_ms_total += retrieval_ms
            self.reranking_latency_ms_total += reranking_ms

    def record_ask(self, retrieval_ms: float, reranking_ms: float, generation_ms: float) -> None:
        with self.lock:
            self.asks_total += 1
            self.retrieval_latency_ms_total += retrieval_ms
            self.reranking_latency_ms_total += reranking_ms
            self.generation_latency_ms_total += generation_ms

    def record_error(self, error: Exception) -> None:
        with self.lock:
            self.errors_total += 1
            self.last_error = str(error)

    def snapshot(self) -> dict:
        with self.lock:
            retrieval_calls = self.searches_total + self.asks_total
            avg_retrieval = (
                self.retrieval_latency_ms_total / retrieval_calls
                if retrieval_calls
                else 0.0
            )
            avg_generation = (
                self.generation_latency_ms_total / self.asks_total
                if self.asks_total
                else 0.0
            )
            avg_reranking = (
                self.reranking_latency_ms_total / retrieval_calls
                if retrieval_calls
                else 0.0
            )
            return {
                "searches_total": self.searches_total,
                "asks_total": self.asks_total,
                "errors_total": self.errors_total,
                "avg_retrieval_latency_ms": round(avg_retrieval, 2),
                "avg_reranking_latency_ms": round(avg_reranking, 2),
                "avg_generation_latency_ms": round(avg_generation, 2),
                "last_error": self.last_error,
            }


class RAGService:
    def __init__(self, load_models: bool = True):
        started = time.perf_counter()
        self.metrics = ServiceMetrics()
        self.products = load_products()
        self.bm25, _ = load_bm25()
        self.index, _ = load_semantic_index()
        self.embedding_model = (
            SentenceTransformer("all-MiniLM-L6-v2") if load_models else None
        )
        self.reranker_model = load_reranker() if load_models else None
        log.info(
            "RAG service loaded products=%s startup_ms=%.2f",
            len(self.products),
            (time.perf_counter() - started) * 1000,
        )

    def health(self) -> dict:
        return {
            "status": "ok",
            "products_loaded": len(self.products),
            "bm25_loaded": self.bm25 is not None,
            "faiss_loaded": self.index is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "reranker_loaded": self.reranker_model is not None,
        }

    def get_metrics(self) -> dict:
        metrics = self.metrics.snapshot()
        metrics["products_loaded"] = len(self.products)
        return metrics

    def search(self, query: str, top_k: int = 5, mode: str = "rerank") -> dict:
        started = time.perf_counter()
        try:
            guardrail = check_input(query)
            if not guardrail["valid"]:
                raise ValueError(guardrail["reason"])

            results, retrieval_ms, reranking_ms = self._retrieve_with_latency(
                query=query,
                top_k=top_k,
                mode=mode,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            self.metrics.record_search(retrieval_ms, reranking_ms)
            log.info(
                "retrieval query=%r mode=%s top_k=%s results=%s guardrail=allow retrieval_ms=%.2f reranking_ms=%.2f latency_ms=%.2f",
                query,
                mode,
                top_k,
                len(results),
                retrieval_ms,
                reranking_ms,
                latency_ms,
            )
            return {
                "query": query,
                "mode": mode,
                "results": [compact_product(result) for result in results],
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "reranking_latency_ms": round(reranking_ms, 2),
                "latency_ms": round(latency_ms, 2),
            }
        except Exception as exc:
            self.metrics.record_error(exc)
            log.exception("search failed query=%r mode=%s", query, mode)
            raise

    def ask(self, query: str, top_k: int = 5) -> dict:
        started = time.perf_counter()
        try:
            result = rag_answer(
                query=query,
                bm25=self.bm25,
                index=self.index,
                products=self.products,
                model=self.embedding_model,
                top_k=top_k,
                reranker_model=self.reranker_model,
            )
            total_ms = (time.perf_counter() - started) * 1000
            generation_ms = float(result.get("generation_latency_ms", 0.0))
            retrieval_ms = float(result.get("retrieval_latency_ms", 0.0))
            reranking_ms = float(result.get("reranking_latency_ms", 0.0))
            self.metrics.record_ask(retrieval_ms, reranking_ms, generation_ms)
            log.info(
                "generation query=%r top_k=%s retrieved=%s guardrail=%s retrieval_ms=%.2f reranking_ms=%.2f total_ms=%.2f generation_ms=%.2f",
                query,
                top_k,
                len(result.get("retrieved", [])),
                result.get("guardrail_decision", "unknown"),
                retrieval_ms,
                reranking_ms,
                total_ms,
                generation_ms,
            )
            result["latency_ms"] = round(total_ms, 2)
            result["retrieved"] = [
                compact_product(product) for product in result.get("retrieved", [])
            ]
            result["sources"] = result["retrieved"]
            return result
        except Exception as exc:
            self.metrics.record_error(exc)
            log.exception("ask failed query=%r", query)
            raise

    def _retrieve(self, query: str, top_k: int, mode: str) -> list[dict]:
        results, _, _ = self._retrieve_with_latency(query=query, top_k=top_k, mode=mode)
        return results

    def _retrieve_with_latency(self, query: str, top_k: int, mode: str) -> tuple[list[dict], float, float]:
        if mode == "bm25":
            started = time.perf_counter()
            return search_bm25(self.bm25, self.products, query, top_k=top_k), (time.perf_counter() - started) * 1000, 0.0
        if mode == "semantic":
            started = time.perf_counter()
            results = search_semantic(
                self.index,
                self.products,
                query,
                top_k=top_k,
                model=self.embedding_model,
            )
            return results, (time.perf_counter() - started) * 1000, 0.0
        if mode == "hybrid":
            started = time.perf_counter()
            results = hybrid_search(
                self.bm25,
                self.index,
                self.products,
                query,
                top_k=top_k,
                model=self.embedding_model,
            )
            return results, (time.perf_counter() - started) * 1000, 0.0
        if mode == "rerank":
            retrieval_started = time.perf_counter()
            candidates = hybrid_search(
                self.bm25,
                self.index,
                self.products,
                query,
                top_k=max(top_k * 4, top_k),
                model=self.embedding_model,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            rerank_started = time.perf_counter()
            results = rerank_products(
                query,
                candidates,
                top_k=top_k,
                model=self.reranker_model,
            )
            reranking_ms = (time.perf_counter() - rerank_started) * 1000
            return results, retrieval_ms, reranking_ms
        raise ValueError("mode must be one of: bm25, semantic, hybrid, rerank")
