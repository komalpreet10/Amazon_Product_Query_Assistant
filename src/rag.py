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

import logging
import time
from functools import lru_cache
from dotenv import load_dotenv
from openai import OpenAI
from src.hybrid import hybrid_search
from src.tools import apply_tools
from src.guardrails import check_input, check_output
from src.reranker import rerank_products
from src.citations import build_citations

load_dotenv()
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
TOP_K      = 5
LLM_MODEL  = "gpt-4o-mini"


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Creates the OpenAI client only when generation is requested.
    """
    return OpenAI()


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
Use bracket citations like [1] or [2] for every product recommendation, matching the product numbers below.
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
    reranker_model=None,
    use_reranker: bool = True,
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
    started = time.perf_counter()

    # input guardrail
    input_check = check_input(query)
    if not input_check["valid"]:
        log.warning("Input guardrail blocked: %s", input_check["reason"])
        return {
            "response": input_check["reason"],
            "retrieved": [],
            "filters": {},
            "context": "",
            "citations": [],
            "guardrail_decision": "blocked",
            "retrieval_latency_ms": 0.0,
            "reranking_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
        }

    # step 1 — retrieve RRF candidates
    candidate_k = max(top_k * 4, top_k) if use_reranker else top_k
    retrieval_started = time.perf_counter()
    retrieved = hybrid_search(bm25, index, products, query, top_k=candidate_k, model=model)
    retrieval_latency_ms = (time.perf_counter() - retrieval_started) * 1000
    log.info("Retrieved %s hybrid candidates", len(retrieved))

    # step 2 — rerank RRF candidates
    reranking_latency_ms = 0.0
    if use_reranker and retrieved:
        reranking_started = time.perf_counter()
        retrieved = rerank_products(query, retrieved, top_k=top_k, model=reranker_model)
        reranking_latency_ms = (time.perf_counter() - reranking_started) * 1000
        log.info("Reranked to %s products", len(retrieved))

    # step 3 — apply tools (price/rating filters)
    filtered, filters = apply_tools(retrieved, query)

    # fall back to retrieved if filters remove everything
    if not filtered:
        log.warning("Filters removed all products — using unfiltered results")
        filtered = retrieved

    # step 4 — build context + prompt
    context = build_context(filtered[:top_k])
    prompt  = build_prompt(query, context, filters)
    citations = build_citations(filtered[:top_k])

    if not filtered or not context.strip():
        log.warning("Generation skipped because retrieved context is empty")
        return {
            "response": "I could not find enough relevant product context to answer reliably. Try a more specific beauty or personal care query.",
            "retrieved": [],
            "filters": filters,
            "context": context,
            "citations": citations,
            "guardrail_decision": "allowed_weak_context",
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "reranking_latency_ms": round(reranking_latency_ms, 2),
            "generation_latency_ms": 0.0,
        }

    # step 5 — generate answer
    generation_started = time.perf_counter()
    try:
        response = get_openai_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
    except Exception as exc:
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000
        log.exception("LLM generation failed: %s", exc)
        return {
            "response": "I found relevant products, but answer generation failed. Review the sources below for the most relevant matches.",
            "retrieved": filtered[:top_k],
            "filters": filters,
            "context": context,
            "citations": citations,
            "guardrail_decision": "allowed_generation_fallback",
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "reranking_latency_ms": round(reranking_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
        }
    generation_latency_ms = (time.perf_counter() - generation_started) * 1000

    answer = response.choices[0].message.content.strip()
    log.info(
        "Answer generated. retrieved=%s generation_latency_ms=%.2f total_latency_ms=%.2f",
        len(filtered[:top_k]),
        generation_latency_ms,
        (time.perf_counter() - started) * 1000,
    )

    # output guardrail
    output_check = check_output(answer, filtered[:top_k])
    if not output_check["valid"]:
        log.warning("Output guardrail blocked: %s", output_check["reason"])
        return {
            "response": "I couldn't generate a reliable answer. Please try rephrasing your query.",
            "retrieved": filtered[:top_k],
            "filters": filters,
            "context": context,
            "citations": citations,
            "guardrail_decision": "blocked_output",
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "reranking_latency_ms": round(reranking_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
        }

    return {
        "response":  answer,
        "retrieved": filtered[:top_k],
        "filters":   filters,
        "context":   context,
        "citations": citations,
        "guardrail_decision": "allowed",
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "reranking_latency_ms": round(reranking_latency_ms, 2),
        "generation_latency_ms": round(generation_latency_ms, 2),
    }
