"""
schemas.py
----------
Pydantic request and response contracts for the production FastAPI service.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


SearchMode = Literal["bm25", "semantic", "hybrid", "rerank"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: SearchMode = "rerank"

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class AskRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class ProductSource(BaseModel):
    parent_asin: str | None = None
    title: str | None = None
    store: str | None = None
    price: float | None = None
    average_rating: float | None = None
    rating_number: int | None = None
    bm25_score: float | None = None
    semantic_score: float | None = None
    hybrid_score: float | None = None
    hybrid_rank: int | None = None
    original_hybrid_rank: int | None = None
    rerank_score: float | None = None
    rerank_position: int | None = None
    snippet: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[ProductSource]
    retrieval_latency_ms: float
    reranking_latency_ms: float
    latency_ms: float


class AskResponse(BaseModel):
    response: str
    sources: list[ProductSource]
    retrieved: list[ProductSource]
    filters: dict
    citations: list[dict]
    guardrail_decision: str
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    latency_ms: float


class MetricsResponse(BaseModel):
    searches_total: int
    asks_total: int
    errors_total: int
    avg_retrieval_latency_ms: float
    avg_reranking_latency_ms: float
    avg_generation_latency_ms: float
    last_error: str | None = None
    products_loaded: int
