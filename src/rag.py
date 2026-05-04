"""
rag.py
------
RAG pipeline for Amazon Product Query Assistant.
Retrieves relevant products, applies filters, and generates
a grounded answer using GPT-4o-mini.

Usage:
    from src.rag import rag_answer

    answer = rag_answer(
        query="best moisturizer for sensitive skin under $30",
        bm25=bm25,
        index=index,
        products=products,
        model=model,
    )
    print(answer["response"])
"""

import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from src.hybrid import hybrid_search
from src.tools import apply_tools

load_dotenv()
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
TOP_K      = 5
LLM_MODEL  = "gpt-4o-mini"
client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ── context builder ────────────────────────────────────────────────────────

def build_context(products: list[dict]) -> str:
    """
    Formats retrieved products into a context string for the LLM prompt.

    Args:
        products: List of top K product dicts.

    Returns:
        Formatted context string.
    """
    context_parts = []

    for i, p in enumerate(products, start=1):
        lines = [f"Product {i}: {p.get('title', 'N/A')}"]

        if p.get("store"):
            lines.append(f"  Store: {p['store']}")
        if p.get("price"):
            lines.append(f"  Price: ${p['price']}")
        if p.get("average_rating"):
            lines.append(f"  Rating: {p['average_rating']} ({p.get('rating_number', 0)} reviews)")
        if p.get("features"):
            lines.append(f"  Features: {' | '.join(p['features'][:3])}")
        if p.get("top_reviews"):
            review = p["top_reviews"][0]
            lines.append(f"  Top Review: {review.get('title', '')} — {review.get('text', '')[:150]}")

        context_parts.append("\n".join(lines))

    return "\n\n".join(context_parts)


# ── prompt builder ─────────────────────────────────────────────────────────

def build_prompt(query: str, context: str, filters: dict) -> str:
    """
    Builds the RAG prompt combining query, context, and filter info.

    Args:
        query:   User query string.
        context: Formatted product context.
        filters: Detected filters (price, rating).

    Returns:
        Prompt string for the LLM.
    """
    filter_note = ""
    if filters.get("max_price"):
        filter_note += f" Price filter applied: under ${filters['max_price']}."
    if filters.get("min_rating"):
        filter_note += f" Rating filter applied: above {filters['min_rating']} stars."

    prompt = f"""You are a helpful Amazon product assistant. Answer the user's query based ONLY on the products provided below. 
Do not make up information or recommend products not in the list.
Be concise, helpful, and specific. Mention product names, prices, and ratings when available.
{filter_note}

User Query: {query}

Retrieved Products:
{context}

Answer:"""

    return prompt


# ── RAG pipeline ───────────────────────────────────────────────────────────

def rag_answer(
    query: str,
    bm25,
    index,
    products: list[dict],
    model,
    top_k: int = TOP_K,
) -> dict:
    """
    Full RAG pipeline: retrieve → filter → generate answer.

    Args:
        query:    User query string.
        bm25:     BM25Okapi index.
        index:    FAISS index.
        products: List of all product dicts.
        model:    SentenceTransformer model.
        top_k:    Number of products to retrieve.

    Returns:
        Dict with:
            - response:  Generated answer from LLM
            - retrieved: List of retrieved products
            - filters:   Detected filters
            - context:   Context string passed to LLM
    """
    log.info("RAG query: %s", query)

    # step 1 — retrieve top K products
    retrieved = hybrid_search(bm25, index, products, query, top_k=top_k, model=model)
    log.info("Retrieved %s products", len(retrieved))

    # step 2 — apply tools (price/rating filters)
    filtered, filters = apply_tools(retrieved, query)

    # fall back to retrieved if filters remove everything
    if not filtered:
        log.warning("Filters removed all products — using unfiltered results")
        filtered = retrieved

    # step 3 — build context + prompt
    context = build_context(filtered[:top_k])
    prompt  = build_prompt(query, context, filters)

    # step 4 — generate answer
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    log.info("Answer generated.")

    return {
        "response":  answer,
        "retrieved": filtered[:top_k],
        "filters":   filters,
        "context":   context,
    }