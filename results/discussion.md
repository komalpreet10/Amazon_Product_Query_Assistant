# Retrieval Evaluation Discussion
## Amazon Product Query Assistant

---

## 1. Overview

This project builds an end-to-end product search assistant over the Amazon Reviews 2023 dataset (All Beauty category). The dataset contains 112,590 products and 701,528 reviews. We implemented and evaluated three retrieval methods: BM25 (keyword-based), Semantic Search (embedding-based), and Hybrid Search (combining both).

---

## 2. Dataset

- **Source:** McAuley Lab Amazon Reviews 2023
- **Category:** All Beauty
- **Products:** 112,590
- **Reviews:** 701,528
- **Key observation:** 85% of products have empty features and 83% have empty descriptions. Reviews are the primary source of rich text content, with 90.5% being verified purchases.

---

## 3. Retrieval Methods

### 3.1 BM25

BM25 is a keyword-based ranking algorithm that scores documents based on term frequency and inverse document frequency, with saturation and document length normalization.

**Tokenization:** Lowercase → punctuation removal → stopword removal → whitespace tokenize

**Strengths:**
- Fast and lightweight
- Works well for exact keyword queries
- No model required

**Weaknesses:**
- Cannot understand meaning or intent
- Fails for vague or semantic queries
- Sensitive to exact word choice

### 3.2 Semantic Search

Semantic search uses the `all-MiniLM-L6-v2` sentence transformer model to embed product search text into 384-dimensional vectors. Queries are embedded at search time and compared against product vectors using cosine similarity via a FAISS IndexFlatIP index.

**Strengths:**
- Understands query intent and meaning
- Works well for vague or natural language queries
- Handles synonyms and related concepts

**Weaknesses:**
- Slower than BM25 (requires embedding query at search time)
- Can miss specific keyword matches
- Requires GPU for fast embedding generation

### 3.3 Hybrid Search (RRF)

Hybrid search combines BM25 and semantic search results using Reciprocal Rank Fusion (RRF). Each product receives a score based on its rank in both result lists:

```
RRF score = 1/(rank_bm25 + 60) + 1/(rank_semantic + 60)
```

Products appearing in both lists get boosted. The constant 60 reduces the impact of very high rankings.

**Strengths:**
- Best of both worlds
- More robust across different query types
- Highest overall Precision@5

---

## 4. Evaluation

### 4.1 Setup

- **Queries:** 10 queries spanning easy (keyword), medium (semantic), and complex difficulty levels
- **Ground truth:** Manually labeled relevant products for each query
- **Metrics:** Precision@5, MRR, NDCG@5

### 4.2 Queries

| # | Query | Type |
|---|---|---|
| 1 | moisturizer for sensitive skin | Easy |
| 2 | vitamin c serum | Easy |
| 3 | shampoo for curly hair | Easy |
| 4 | something to keep my skin hydrated all day | Medium |
| 5 | product for damaged hair | Medium |
| 6 | gentle face wash for acne | Medium |
| 7 | best affordable moisturizer under $20 for dry skin | Complex |
| 8 | natural organic hair care for color treated hair | Complex |
| 9 | anti aging cream for women over 50 | Complex |
| 10 | fragrance free products for baby sensitive skin | Complex |

### 4.3 Results

| Method | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|
| BM25 | 0.60 | 0.75 | 0.60 |
| Semantic | 0.66 | 0.83 | 0.69 |
| **Hybrid** | **0.78** | 0.73 | **0.71** |

---

## 5. Observations

### 5.1 Where BM25 wins

BM25 performs well for exact keyword queries like "vitamin c serum" and "moisturizer for sensitive skin" where the query words appear directly in product titles. It consistently finds the first relevant result quickly (MRR = 0.75).

### 5.2 Where Semantic wins

Semantic search significantly outperforms BM25 for vague intent-based queries. For example:

- **Query:** "something gentle for my skin"
  - BM25 returned a bath brush and a retinol serum — matched "gentle" literally
  - Semantic returned sensitive skin cleansers and lotions — understood intent

- **Query:** "product for damaged hair"
  - BM25 matched "damaged" literally but returned irrelevant products
  - Semantic understood "damaged hair" = repair treatments and returned relevant results

Semantic search has the highest MRR (0.83), meaning it finds the first relevant result faster than both other methods.

### 5.3 Where Hybrid wins

Hybrid search achieves the best Precision@5 (0.78) and NDCG@5 (0.71) overall. By combining both methods, it handles both keyword-specific and intent-based queries well. Products appearing in both BM25 and semantic result lists get boosted, which consistently improves result quality.

### 5.4 Failure cases

- **Price filtering:** "best affordable moisturizer under $20" — none of the methods can reliably filter by price since 84% of products have missing price data.
- **Specificity:** "anti aging cream for women over 50" — none of the methods explicitly handle demographic targeting. Results are general anti-aging creams.
- **Wrong category:** BM25 occasionally returns irrelevant products (e.g., a hair dryer for a moisturizer query) when query words match unrelated product descriptions.

---

## 6. Conclusions

### Strengths and Weaknesses

| Method | Best For | Worst For |
|---|---|---|
| BM25 | Exact keyword queries, fast retrieval | Vague queries, synonyms, intent |
| Semantic | Intent-based queries, natural language | Specific attributes, price queries |
| Hybrid | General use, best overall performance | Price/demographic filtering |

### What requires more advanced methods

1. **Price filtering** — needs a dedicated filter layer applied before/after retrieval
2. **RAG** — LLM response generation will help synthesize results into helpful answers
3. **Guardrails** — needed to block off-topic queries and validate LLM outputs
4. **Reranking** — a cross-encoder reranker could further improve result ordering

---

## 7. Next Steps

- **RAG Pipeline:** Use retrieved products as context for LLM-generated responses
- **Tool Integration:** Add price filtering, rating filtering as tools
- **Guardrails:** Input validation and output grounding checks
- **RAGAS Evaluation:** Evaluate faithfulness and answer relevance of generated responses
