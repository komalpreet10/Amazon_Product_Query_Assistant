"""
guardrails.py
-------------
Input and output guardrails for the Amazon Product Query Assistant.

Input guardrails:  validate user query before processing
Output guardrails: validate LLM answer before showing to user

Usage:
    from src.guardrails import check_input, check_output

    # check query
    result = check_input("best moisturizer for sensitive skin")
    if not result["valid"]:
        print(result["reason"])

    # check answer
    result = check_output(answer, retrieved_products)
    if not result["valid"]:
        print(result["reason"])
"""

import re
import logging

log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────

# beauty/personal care related keywords
BEAUTY_KEYWORDS = [
    "skin", "hair", "face", "moisturizer", "serum", "shampoo", "conditioner",
    "cream", "lotion", "cleanser", "sunscreen", "spf", "acne", "wrinkle",
    "aging", "dry", "oily", "sensitive", "fragrance", "lip", "eye", "body",
    "wash", "scrub", "mask", "toner", "oil", "balm", "gel", "foam", "soap",
    "deodorant", "perfume", "cologne", "makeup", "foundation", "blush",
    "mascara", "lipstick", "nail", "beard", "shaving", "baby", "organic",
    "natural", "vitamin", "collagen", "hyaluronic", "retinol", "spf"
]

# harmful keywords to block
HARMFUL_KEYWORDS = [
    "bomb", "weapon", "kill", "harm", "poison", "drug", "illegal",
    "hack", "attack", "violence", "suicide", "self-harm",
]

# medical claim keywords to block in output
MEDICAL_CLAIM_KEYWORDS = [
    "cures", "cure", "treats", "treatment for", "heals", "healing",
    "clinically proven", "fda approved", "eliminates", "prevents cancer",
    "prevents disease", "medical treatment",
]

MIN_QUERY_LENGTH  = 3    # minimum words in query
MAX_QUERY_LENGTH  = 50   # maximum words in query
MIN_ANSWER_LENGTH = 20   # minimum words in answer


# ── input guardrails ───────────────────────────────────────────────────────

def check_input(query: str) -> dict:
    """
    Validates user query before processing.

    Checks:
        1. Not empty
        2. Not too short or too long
        3. Not gibberish
        4. Not harmful
        5. Related to beauty/personal care

    Args:
        query: User query string.

    Returns:
        Dict with 'valid' (bool) and 'reason' (str if invalid).
    """
    # 1. empty check
    if not query or not query.strip():
        return {"valid": False, "reason": "Query cannot be empty."}

    query = query.strip()
    words = query.split()

    # 2. length check
    if len(words) < MIN_QUERY_LENGTH:
        return {"valid": False, "reason": "Query is too short. Please be more specific."}

    if len(words) > MAX_QUERY_LENGTH:
        return {"valid": False, "reason": "Query is too long. Please keep it under 50 words."}

    # 3. gibberish check — too many non-alphabetic characters
    alpha_ratio = sum(c.isalpha() for c in query) / len(query)
    if alpha_ratio < 0.5:
        return {"valid": False, "reason": "Query appears to be gibberish. Please enter a valid question."}

    # 4. harmful content check
    query_lower = query.lower()
    for keyword in HARMFUL_KEYWORDS:
        if keyword in query_lower:
            return {"valid": False, "reason": "Query contains inappropriate content."}

    # 5. off-topic check — must contain at least one beauty keyword
    if not any(keyword in query_lower for keyword in BEAUTY_KEYWORDS):
        return {
            "valid": False,
            "reason": "This assistant only answers questions about beauty and personal care products. Please ask about skincare, haircare, or similar topics."
        }

    log.info("Input guardrail passed: %s", query)
    return {"valid": True, "reason": None}


# ── output guardrails ──────────────────────────────────────────────────────

def check_output(answer: str, retrieved: list[dict]) -> dict:
    """
    Validates LLM answer before showing to user.

    Checks:
        1. Not empty or too short
        2. No medical claims
        3. Grounded in retrieved products

    Args:
        answer:    LLM generated answer string.
        retrieved: List of retrieved product dicts.

    Returns:
        Dict with 'valid' (bool) and 'reason' (str if invalid).
    """
    if not answer or not answer.strip():
        return {"valid": False, "reason": "Empty answer generated."}

    words = answer.strip().split()

    # 1. minimum length check
    if len(words) < MIN_ANSWER_LENGTH:
        return {"valid": False, "reason": "Answer is too short or uninformative."}

    # 2. medical claims check
    answer_lower = answer.lower()
    for keyword in MEDICAL_CLAIM_KEYWORDS:
        if keyword in answer_lower:
            return {
                "valid": False,
                "reason": "Answer contains medical claims which cannot be verified."
            }

    # 3. grounding check — answer should mention at least one retrieved product title
    product_titles = [p.get("title", "").lower()[:30] for p in retrieved]
    grounded = any(
        title[:15] in answer_lower
        for title in product_titles
        if title
    )
    if not grounded:
        return {
            "valid": False,
            "reason": "Answer is not grounded in retrieved products."
        }

    log.info("Output guardrail passed.")
    return {"valid": True, "reason": None}