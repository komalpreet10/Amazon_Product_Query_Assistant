"""
semantic.py
-----------
Semantic retrieval system using sentence-transformers and FAISS.
Builds, persists, and queries a FAISS index over product embeddings.

Usage:
    from src.semantic import build_semantic_index, search_semantic, load_semantic_index

    # build once
    index, embeddings = build_semantic_index(corpus, products)

    # load later
    index, embeddings = load_semantic_index()

    # search
    results = search_semantic(index, products, "moisturizer for sensitive skin", top_k=5)
"""

import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── logging ────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── config ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = Path("data/processed/faiss.index")
EMBEDDINGS_PATH  = Path("data/processed/embeddings.npy")
BATCH_SIZE       = 256


# ── build ──────────────────────────────────────────────────────────────────

def build_semantic_index(
    corpus: list[str],
    faiss_path: Path = FAISS_INDEX_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> tuple:
    """
    Embeds the corpus using all-MiniLM-L6-v2 and builds a FAISS index.

    Args:
        corpus:          List of raw search_text strings (one per product).
        faiss_path:      Where to save the FAISS index.
        embeddings_path: Where to save the raw embeddings.

    Returns:
        Tuple of (faiss_index, embeddings)
    """
    log.info("Loading model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    log.info("Embedding %s documents (batch_size=%s)...", f"{len(corpus):,}", BATCH_SIZE)
    embeddings = model.encode(
        corpus,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # normalize for cosine similarity
    )

    log.info("Embeddings shape: %s", embeddings.shape)

    # build FAISS index (inner product = cosine similarity since normalized)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("FAISS index built: %s vectors", index.ntotal)

    # save
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_path))
    np.save(embeddings_path, embeddings)
    log.info("FAISS index saved → %s", faiss_path)
    log.info("Embeddings saved  → %s", embeddings_path)

    return index, embeddings


# ── load ───────────────────────────────────────────────────────────────────

def load_semantic_index(
    faiss_path: Path      = FAISS_INDEX_PATH,
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> tuple:
    """
    Loads a persisted FAISS index and embeddings from disk.

    Args:
        faiss_path:      Path to saved FAISS index.
        embeddings_path: Path to saved embeddings.

    Returns:
        Tuple of (faiss_index, embeddings)
    """
    log.info("Loading FAISS index from %s...", faiss_path)
    index      = faiss.read_index(str(faiss_path))
    embeddings = np.load(embeddings_path)
    log.info("FAISS index loaded: %s vectors", index.ntotal)
    return index, embeddings


# ── search ─────────────────────────────────────────────────────────────────

def search_semantic(
    index: faiss.Index,
    products: list[dict],
    query: str,
    top_k: int = 5,
    model: SentenceTransformer = None,
) -> list[dict]:
    """
    Searches the FAISS index for a query and returns top K products.

    Args:
        index:    FAISS index.
        products: List of product dicts (same order as corpus).
        query:    Natural language query string.
        top_k:    Number of results to return.
        model:    SentenceTransformer model (loaded if not provided).

    Returns:
        List of top K product dicts with an added 'semantic_score' field.
    """
    if model is None:
        model = SentenceTransformer(MODEL_NAME)

    # embed query (normalize for cosine similarity)
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # search FAISS index
    scores, indices = index.search(query_vector, top_k)

    # build results
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        product = products[idx].copy()
        product["semantic_score"] = round(float(score), 4)
        results.append(product)

    return results