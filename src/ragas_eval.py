"""
ragas_eval.py
-------------
RAGAS evaluation for the Amazon Product Query Assistant RAG pipeline.
Evaluates faithfulness and answer relevancy using GPT as judge.

Usage:
    from src.ragas_eval import run_ragas_evaluation

    scores = run_ragas_evaluation(queries, bm25, index, products, model)
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.rag import rag_answer

load_dotenv()
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("results/ragas_results.json")


def run_ragas_evaluation(
    queries: list[str],
    bm25,
    index,
    products: list[dict],
    model,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """
    Runs RAGAS evaluation on a list of queries.

    Steps:
        1. Run RAG pipeline for each query
        2. Build evaluation dataset
        3. Evaluate with RAGAS faithfulness + answer relevancy
        4. Save results to disk

    Args:
        queries:     List of evaluation queries.
        bm25:        BM25Okapi index.
        index:       FAISS index.
        products:    List of product dicts.
        model:       SentenceTransformer model.
        output_path: Where to save results.

    Returns:
        Dict with RAGAS scores per query and averages.
    """
    log.info("Running RAGAS evaluation on %s queries...", len(queries))

    # step 1 — run RAG pipeline for each query
    data = {"question": [], "answer": [], "contexts": []}

    for query in queries:
        log.info("Evaluating query: %s", query)
        result = rag_answer(query, bm25, index, products, model)

        # skip blocked queries
        if not result["retrieved"]:
            log.warning("Skipping blocked query: %s", query)
            continue

        data["question"].append(query)
        data["answer"].append(result["response"])
        data["contexts"].append([result["context"]])

    if not data["question"]:
        log.error("No valid queries to evaluate.")
        return {}

    # step 2 — build HuggingFace dataset
    dataset = Dataset.from_dict(data)
    log.info("Evaluation dataset built: %s samples", len(dataset))

    # step 3 — set up LLM and embeddings for RAGAS
    llm        = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    faithfulness.llm            = llm
    answer_relevancy.llm        = llm
    answer_relevancy.embeddings = embeddings

    # step 4 — run RAGAS evaluation
    log.info("Running RAGAS evaluation...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    # step 5 — format and save results
    results = {
        "scores_per_query": scores.to_pandas().to_dict(orient="records"),
        "average": {
            "faithfulness":     round(float(np.nanmean(scores["faithfulness"])), 4),
            "answer_relevancy": round(float(np.nanmean(scores["answer_relevancy"])), 4),
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("RAGAS results saved → %s", output_path)

    print("\n── RAGAS Evaluation Results ────────────────")
    print(f"  Faithfulness:     {results['average']['faithfulness']}")
    print(f"  Answer Relevancy: {results['average']['answer_relevancy']}")
    print("────────────────────────────────────────────\n")

    return results