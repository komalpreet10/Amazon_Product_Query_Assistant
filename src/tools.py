"""
tools.py
--------
Filter tools for the RAG pipeline.
Used by the LLM to filter products based on price and rating.

Usage:
    from src.tools import filter_by_price, filter_by_rating, apply_tools
"""

import logging
import re

log = logging.getLogger(__name__)


# ── filter tools ───────────────────────────────────────────────────────────

def filter_by_price(products: list[dict], max_price: float) -> list[dict]:
    """
    Filters products to those at or below max_price.
    Skips products with missing price.

    Args:
        products:  List of product dicts.
        max_price: Maximum price in USD.

    Returns:
        Filtered list of products.
    """
    filtered = [
        p for p in products
        if p.get("price") is not None and p["price"] <= max_price
    ]
    log.info("filter_by_price(max=%.2f): %s → %s products", max_price, len(products), len(filtered))
    return filtered


def filter_by_rating(products: list[dict], min_rating: float) -> list[dict]:
    """
    Filters products to those at or above min_rating.
    Skips products with missing rating.

    Args:
        products:   List of product dicts.
        min_rating: Minimum average rating (1.0 to 5.0).

    Returns:
        Filtered list of products.
    """
    filtered = [
        p for p in products
        if p.get("average_rating") is not None and p["average_rating"] >= min_rating
    ]
    log.info("filter_by_rating(min=%.1f): %s → %s products", min_rating, len(products), len(filtered))
    return filtered


# ── tool detector ──────────────────────────────────────────────────────────

def detect_filters(query: str) -> dict:
    """
    Detects price and rating filters from a natural language query.

    Args:
        query: User query string.

    Returns:
        Dict with detected filters e.g. {"max_price": 20.0, "min_rating": 4.0}

    Examples:
        "moisturizer under $20"           → {"max_price": 20.0}
        "rated above 4 stars"             → {"min_rating": 4.0}
        "best serum under $30 rating 4.5" → {"max_price": 30.0, "min_rating": 4.5}
    """
    filters = {}

    # detect price — matches "under $20", "less than $30", "below 25"
    price_match = re.search(
        r"(?:under|less than|below|max|cheaper than)\s*\$?(\d+(?:\.\d+)?)",
        query, re.IGNORECASE
    )
    if price_match:
        filters["max_price"] = float(price_match.group(1))

    # detect rating — matches "above 4 stars", "rating 4.5", "at least 4"
    rating_match = re.search(
        r"(?:above|over|at least|minimum|rated|rating)\s*(\d+(?:\.\d+)?)\s*(?:stars?)?",
        query, re.IGNORECASE
    )
    if rating_match:
        rating = float(rating_match.group(1))
        if 1.0 <= rating <= 5.0:
            filters["min_rating"] = rating

    if filters:
        log.info("Detected filters: %s", filters)

    return filters


# ── apply tools ────────────────────────────────────────────────────────────

def apply_tools(products: list[dict], query: str) -> tuple[list[dict], dict]:
    """
    Detects and applies filters from the query to the product list.

    Args:
        products: List of product dicts.
        query:    User query string.

    Returns:
        Tuple of (filtered_products, detected_filters)
    """
    filters = detect_filters(query)

    if "max_price" in filters:
        products = filter_by_price(products, filters["max_price"])

    if "min_rating" in filters:
        products = filter_by_rating(products, filters["min_rating"])

    return products, filters