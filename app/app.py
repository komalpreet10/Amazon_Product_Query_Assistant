"""
app.py
------
Streamlit app for Amazon Product Query Assistant.
Supports BM25, Semantic, Hybrid, and RAG search modes.

Run:
    streamlit run app/app.py
"""

import json
import logging
from pathlib import Path

import streamlit as st

from src.bm25 import load_bm25, search_bm25
from src.semantic import load_semantic_index, search_semantic
from src.hybrid import hybrid_search
from src.rag import rag_answer
from src.guardrails import check_input
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Assistant",
    page_icon="🛍️",
    layout="wide",
)

# ── custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        color: #1a1a2e;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    .product-card {
        background: #ffffff;
        border: 1px solid #e8e8e8;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s;
    }

    .product-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .product-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }

    .product-meta {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 0.5rem;
    }

    .product-review {
        font-size: 0.85rem;
        color: #555;
        font-style: italic;
        border-left: 3px solid #f0c040;
        padding-left: 0.7rem;
        margin-top: 0.5rem;
    }

    .rag-answer {
        background: linear-gradient(135deg, #f8f4ff, #fff8f0);
        border: 1px solid #e0d0ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #2d2d2d;
    }

    .score-badge {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        color: #666;
        margin-left: 0.5rem;
    }

    .error-box {
        background: #fff0f0;
        border: 1px solid #ffcccc;
        border-radius: 8px;
        padding: 1rem;
        color: #cc0000;
        font-size: 0.9rem;
    }

    .stRadio > div {
        flex-direction: row;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── load resources (cached) ────────────────────────────────────────────────

@st.cache_resource
def load_resources():
    products_path = Path("data/processed/products.jsonl")
    with open(products_path, "r") as f:
        products = [json.loads(line) for line in f if line.strip()]

    bm25, _  = load_bm25()
    index, _ = load_semantic_index()
    model    = SentenceTransformer("all-MiniLM-L6-v2")

    return products, bm25, index, model


# ── helper: render product card ────────────────────────────────────────────

def render_product(product: dict, score: float = None, score_label: str = None):
    rating      = product.get("average_rating", "N/A")
    rating_num  = product.get("rating_number", 0)
    price       = f"${product['price']}" if product.get("price") else "Price N/A"
    store       = product.get("store", "")
    stars       = "⭐" * int(round(rating)) if isinstance(rating, float) else ""
    top_reviews = product.get("top_reviews", [])

    score_html = ""
    if score is not None:
        score_html = f'<span class="score-badge">{score_label}: {score}</span>'

    review_html = ""
    if top_reviews:
        review = top_reviews[0]
        review_text = review.get("text", "")[:200]
        review_html = f'<div class="product-review">"{review_text}..."</div>'

    st.markdown(f"""
    <div class="product-card">
        <div class="product-title">{product.get('title', 'N/A')[:100]} {score_html}</div>
        <div class="product-meta">{stars} {rating} ({rating_num:,} reviews) &nbsp;|&nbsp; {price} &nbsp;|&nbsp; {store}</div>
        {review_html}
    </div>
    """, unsafe_allow_html=True)


# ── main app ───────────────────────────────────────────────────────────────

def main():
    # header
    st.markdown('<h1 class="main-title">🛍️ Amazon Product Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Search 112K beauty products using BM25, Semantic, Hybrid, or RAG</p>', unsafe_allow_html=True)

    # load resources
    with st.spinner("Loading search indexes..."):
        products, bm25, index, model = load_resources()

    # search mode
    mode = st.radio(
        "Search Mode",
        ["BM25", "Semantic", "Hybrid", "RAG"],
        horizontal=True,
    )

    # mode descriptions
    mode_descriptions = {
        "BM25":     "⚡ Keyword-based search — fast and precise for exact terms",
        "Semantic": "🧠 Meaning-based search — understands intent and context",
        "Hybrid":   "🔀 Combines BM25 + Semantic for best overall results",
        "RAG":      "🤖 AI-generated answer grounded in retrieved products",
    }
    st.caption(mode_descriptions[mode])

    # search input
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Search",
            placeholder="e.g. best moisturizer for sensitive skin under $30",
            label_visibility="collapsed",
        )
    with col2:
        search = st.button("Search", use_container_width=True)

    # top K selector
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5)

    # run search
    if search and query:

        # input guardrail
        guard = check_input(query)
        if not guard["valid"]:
            st.markdown(f'<div class="error-box">⚠️ {guard["reason"]}</div>', unsafe_allow_html=True)
            return

        with st.spinner("Searching..."):

            if mode == "BM25":
                results = search_bm25(bm25, products, query, top_k=top_k)
                st.subheader(f"Top {len(results)} BM25 Results")
                for r in results:
                    render_product(r, score=r["bm25_score"], score_label="BM25")

            elif mode == "Semantic":
                results = search_semantic(index, products, query, top_k=top_k, model=model)
                st.subheader(f"Top {len(results)} Semantic Results")
                for r in results:
                    render_product(r, score=r["semantic_score"], score_label="Score")

            elif mode == "Hybrid":
                results = hybrid_search(bm25, index, products, query, top_k=top_k, model=model)
                st.subheader(f"Top {len(results)} Hybrid Results")
                for r in results:
                    render_product(r, score=r["hybrid_score"], score_label="RRF")

            elif mode == "RAG":
                result = rag_answer(query, bm25, index, products, model, top_k=top_k)

                # show generated answer
                st.subheader("🤖 AI Answer")
                st.markdown(f'<div class="rag-answer">{result["response"]}</div>', unsafe_allow_html=True)

                # show filters if applied
                if result["filters"]:
                    st.caption(f"🔧 Filters applied: {result['filters']}")

                # show retrieved products
                if result["retrieved"]:
                    st.subheader("📦 Retrieved Products")
                    for r in result["retrieved"]:
                        render_product(r)


if __name__ == "__main__":
    main()
