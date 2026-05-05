"""
test_retrieval.py
-----------------
Basic unit tests for the Amazon Product Query Assistant.
Tests tokenization, guardrails, tools, and metrics.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import tokenize
from src.guardrails import check_input, check_output
from src.tools import detect_filters, filter_by_price, filter_by_rating
from src.retrieval_metrics import precision_at_k, mrr, ndcg_at_k


# ── tokenizer tests ────────────────────────────────────────────────────────

def test_tokenize_basic():
    tokens = tokenize("Best moisturizer for sensitive skin!")
    assert "moisturizer" in tokens
    assert "sensitive" in tokens
    assert "skin" in tokens

def test_tokenize_removes_stopwords():
    tokens = tokenize("best moisturizer for the skin")
    assert "for" not in tokens
    assert "the" not in tokens

def test_tokenize_lowercase():
    tokens = tokenize("CeraVe Moisturizer")
    assert "cerave" in tokens

def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize(None) == []


# ── guardrail tests ────────────────────────────────────────────────────────

def test_input_valid_query():
    result = check_input("best moisturizer for sensitive skin")
    assert result["valid"] is True

def test_input_empty_query():
    result = check_input("")
    assert result["valid"] is False

def test_input_too_short():
    result = check_input("hi")
    assert result["valid"] is False

def test_input_harmful():
    result = check_input("how to make a bomb")
    assert result["valid"] is False

def test_input_off_topic():
    result = check_input("best laptop under five hundred dollars")
    assert result["valid"] is False

def test_input_gibberish():
    result = check_input("asdfjkl; 123 !!!")
    assert result["valid"] is False


# ── tools tests ────────────────────────────────────────────────────────────

def test_detect_price_filter():
    filters = detect_filters("moisturizer under $20")
    assert filters.get("max_price") == 20.0

def test_detect_rating_filter():
    filters = detect_filters("serum rated above 4 stars")
    assert filters.get("min_rating") == 4.0

def test_detect_no_filter():
    filters = detect_filters("best moisturizer for sensitive skin")
    assert filters == {}

def test_filter_by_price():
    products = [
        {"parent_asin": "A1", "price": 15.0},
        {"parent_asin": "A2", "price": 25.0},
        {"parent_asin": "A3", "price": None},
    ]
    filtered = filter_by_price(products, max_price=20.0)
    assert len(filtered) == 1
    assert filtered[0]["parent_asin"] == "A1"

def test_filter_by_rating():
    products = [
        {"parent_asin": "A1", "average_rating": 4.5},
        {"parent_asin": "A2", "average_rating": 3.2},
        {"parent_asin": "A3", "average_rating": None},
    ]
    filtered = filter_by_rating(products, min_rating=4.0)
    assert len(filtered) == 1
    assert filtered[0]["parent_asin"] == "A1"


# ── metrics tests ──────────────────────────────────────────────────────────

def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant  = ["A", "C", "F"]
    assert precision_at_k(retrieved, relevant, k=5) == 2/5

def test_precision_at_k_no_relevant():
    retrieved = ["A", "B", "C"]
    relevant  = ["X", "Y"]
    assert precision_at_k(retrieved, relevant, k=3) == 0.0

def test_mrr_first_result():
    retrieved = ["A", "B", "C"]
    relevant  = ["A"]
    assert mrr(retrieved, relevant) == 1.0

def test_mrr_second_result():
    retrieved = ["B", "A", "C"]
    relevant  = ["A"]
    assert mrr(retrieved, relevant) == 0.5

def test_mrr_not_found():
    retrieved = ["A", "B", "C"]
    relevant  = ["X"]
    assert mrr(retrieved, relevant) == 0.0

def test_ndcg_at_k_perfect():
    retrieved = ["A", "B", "C"]
    relevant  = ["A", "B", "C"]
    assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

def test_ndcg_at_k_no_relevant():
    retrieved = ["A", "B", "C"]
    relevant  = ["X", "Y"]
    assert ndcg_at_k(retrieved, relevant, k=3) == 0.0