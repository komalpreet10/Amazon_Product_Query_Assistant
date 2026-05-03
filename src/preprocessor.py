"""
preprocessor.py
---------------
Merges raw metadata + reviews on parent_asin,
extracts useful fields, and builds a search_text per product.

Usage:
    from src.preprocessor import build_products
    products = build_products(meta, reviews)
"""

import json
import ast
import logging
from pathlib import Path
from collections import defaultdict

# ── logging ────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
MAX_REVIEWS      = 3
OUTPUT_PATH      = Path("data/processed/products.jsonl")

# detail keys we care about
USEFUL_DETAIL_KEYS = {
    "Brand":             "brand",
    "Item Form":         "item_form",
    "Material":          "material",
    "Hair Type":         "hair_type",
    "Age Range (Description)": "age_range",
}


# ── helpers ────────────────────────────────────────────────────────────────

def _clean_price(value) -> float | None:
    if not value or str(value).strip().lower() in ("none", "", "null"):
        return None
    try:
        return round(float(str(value).replace("$", "").replace(",", "").strip()), 2)
    except (ValueError, TypeError):
        return None

def _parse_details(raw) -> dict:
    """
    Parse details field (comes as string) into a dict.
    Extracts only the useful keys we defined above.
    """
    result = {v: None for v in USEFUL_DETAIL_KEYS.values()}
    if not raw:
        return result
    try:
        parsed = ast.literal_eval(str(raw))
        for raw_key, clean_key in USEFUL_DETAIL_KEYS.items():
            result[clean_key] = parsed.get(raw_key)
    except Exception:
        pass
    return result


def _select_top_reviews(reviews: list, n: int = MAX_REVIEWS) -> list:
    """
    Pick top N reviews by helpful_vote among verified purchases.
    Falls back to all reviews if none are verified.
    """
    verified = [r for r in reviews if r.get("verified_purchase")]
    pool     = verified if verified else reviews
    ranked   = sorted(pool, key=lambda r: r.get("helpful_vote", 0), reverse=True)

    return [
        {
            "title":  str(r.get("title",  "") or "").strip(),
            "text":   str(r.get("text",   "") or "").strip(),
            "rating": r.get("rating"),
        }
        for r in ranked[:n]
        if str(r.get("text", "")).strip()   # skip empty review text
    ]


def _build_search_text(product: dict) -> str:
    """
    Builds a single string per product used by both BM25 and embeddings.
    Combines all useful text fields into one searchable document.
    """
    parts = []

    # title — always present
    if product.get("title"):
        parts.append(product["title"])

    # store
    if product.get("store"):
        parts.append(f"Store: {product['store']}")

    # details
    if product.get("brand"):
        parts.append(f"Brand: {product['brand']}")
    if product.get("item_form"):
        parts.append(f"Form: {product['item_form']}")
    if product.get("material"):
        parts.append(f"Material: {product['material']}")
    if product.get("hair_type"):
        parts.append(f"Hair Type: {product['hair_type']}")
    if product.get("age_range"):
        parts.append(f"Age Range: {product['age_range']}")

    # price
    if product.get("price"):
        parts.append(f"Price: ${product['price']}")

    # features (only 15% have these but very rich when present)
    if product.get("features"):
        parts.append("Features: " + " ".join(product["features"]))

    # description (only 17% have these)
    if product.get("description"):
        parts.append("Description: " + " ".join(product["description"]))

    # top reviews — our richest source
    # top reviews — truncated for search_text only
    if product.get("top_reviews"):
        review_texts = " ".join(
            f"{r['title']}. {r['text'][:200]}"
            for r in product["top_reviews"]
            if r.get("text")
        )
        if review_texts:
            parts.append(f"Reviews: {review_texts}")


    return " | ".join(parts)


# ── main function ──────────────────────────────────────────────────────────

def build_products(
    meta,
    reviews,
    output_path: Path = OUTPUT_PATH,
) -> list[dict]:
    """
    Merges metadata + reviews, builds search_text per product,
    saves to output_path and returns list of product dicts.

    Args:
        meta:        Raw metadata (list of dicts or HuggingFace dataset)
        reviews:     Raw reviews (list of dicts or HuggingFace dataset)
        output_path: Where to save products.jsonl

    Returns:
        List of product dicts, each with a search_text field.
    """

    # ── step 1: group reviews by parent_asin ──────────────────────────
    log.info("Grouping reviews by parent_asin...")
    reviews_by_asin = defaultdict(list)
    for row in reviews:
        asin = row.get("parent_asin")
        if asin:
            reviews_by_asin[asin].append(row)
    log.info("  %s unique products have reviews", f"{len(reviews_by_asin):,}")

    # ── step 2: build one document per product ────────────────────────
    log.info("Building product documents...")
    products = []

    for row in meta:
        asin = row.get("parent_asin")
        if not asin:
            continue

        # extract details
        details = _parse_details(row.get("details"))

        product = {
            # identity
            "parent_asin":    asin,

            # core fields
            "title":          str(row.get("title", "") or "").strip(),
            "store":          str(row.get("store", "") or "").strip(),
            "price":          _clean_price(row.get("price")),
            "average_rating": row.get("average_rating"),
            "rating_number":  row.get("rating_number"),

            # text fields (sparse but useful when present)
            "features":       [str(f).strip() for f in (row.get("features") or []) if f],
            "description":    [str(d).strip() for d in (row.get("description") or []) if d],

            # extracted from details
            "brand":          details["brand"],
            "item_form":      details["item_form"],
            "material":       details["material"],
            "hair_type":      details["hair_type"],
            "age_range":      details["age_range"],

            # top reviews
            "top_reviews":    _select_top_reviews(reviews_by_asin.get(asin, [])),
        }

        # build search text
        product["search_text"] = _build_search_text(product)

        # skip products with no usable search text
        if not product["search_text"].strip():
            continue

        products.append(product)

    log.info("Total products built: %s", f"{len(products):,}")

    # ── step 3: save ──────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in products:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    log.info("Saved → %s", output_path)

    return products