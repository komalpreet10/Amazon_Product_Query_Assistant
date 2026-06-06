"""
api.py
------
FastAPI application exposing production RAG endpoints.
"""

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import (
    AskRequest,
    AskResponse,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
)
from src.service import RAGService


def configure_logging() -> None:
    """
    Configures retrieval logging without recording secrets.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    log_path = log_dir / "retrieval.log"
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path.resolve()
        for handler in root.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)


configure_logging()
log = logging.getLogger(__name__)

app = FastAPI(
    title="Amazon Product Query Assistant API",
    version="1.0.0",
    description="Hybrid retrieval, cross-encoder reranking, and grounded RAG over Amazon product data.",
)


@lru_cache(maxsize=1)
def _build_service() -> RAGService:
    return RAGService()


def get_service() -> RAGService:
    try:
        return _build_service()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready(service: RAGService = Depends(get_service)) -> dict:
    return service.health()


@app.get("/metrics", response_model=MetricsResponse)
def metrics(service: RAGService = Depends(get_service)) -> dict:
    return service.get_metrics()


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, service: RAGService = Depends(get_service)) -> dict:
    try:
        return service.search(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
        )
    except ValueError as exc:
        log.info("search rejected query=%r reason=%s", request.query, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Unhandled search error")
        raise HTTPException(status_code=500, detail="Search failed") from exc


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, service: RAGService = Depends(get_service)) -> dict:
    try:
        return service.ask(query=request.query, top_k=request.top_k)
    except ValueError as exc:
        log.info("generation rejected query=%r reason=%s", request.query, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Unhandled generation error")
        raise HTTPException(status_code=500, detail="Generation failed") from exc
