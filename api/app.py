"""
Compatibility wrapper for the production FastAPI app.

Prefer running:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

from app.api import app, get_service

__all__ = ["app", "get_service"]
