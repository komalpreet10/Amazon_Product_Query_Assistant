# 🛍️ Amazon Product Query Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system built over 112K Amazon Beauty products and 700K reviews. The system supports hybrid retrieval (BM25 + Semantic Search), tool-augmented RAG with GPT-4o-mini, input/output guardrails, and a Streamlit web app.

---

## 🏗️ Architecture

```
User Query
    ↓
Input Guardrails (validate query)
    ↓
Hybrid Retrieval (BM25 + Semantic Search + RRF)
    ↓
Tool Integration (price filter, rating filter)
    ↓
RAG Pipeline (GPT-4o-mini + retrieved context)
    ↓
Output Guardrails (validate answer)
    ↓
Streamlit App (display answer + products)
```

---

## ✨ Features

- **Hybrid Retrieval** — combines BM25 keyword search and semantic search (all-MiniLM-L6-v2 + FAISS) using Reciprocal Rank Fusion (RRF)
- **RAG Pipeline** — GPT-4o-mini generates answers grounded in retrieved product context
- **Tool Integration** — automatic price and rating filter detection from natural language queries
- **Guardrails** — input validation (off-topic, harmful, gibberish detection) and output validation (medical claims, grounding checks)
- **Evaluation** — retrieval quality (Precision@5, MRR, NDCG@5) and RAG quality (RAGAS faithfulness, answer relevancy)
- **Streamlit App** — interactive web app with BM25 / Semantic / Hybrid / RAG search modes
- **Docker + CI/CD** — containerized with Docker, automated testing with GitHub Actions

---

## 📊 Evaluation Results

### Retrieval Metrics

| Method | Precision@5 | MRR | NDCG@5 |
|--------|-------------|-----|--------|
| BM25 | 0.60 | 0.75 | 0.60 |
| Semantic | 0.66 | 0.83 | 0.69 |
| **Hybrid** | **0.78** | 0.73 | **0.71** |

### RAGAS Metrics

| Metric | Score |
|--------|-------|
| Faithfulness | 0.57 |
| Answer Relevancy | 0.60 |

---

## 📁 Project Structure

```
Amazon_Product_Query_Assistant/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .env                          
├── .gitignore
│
├── data/
│   ├── raw/                      # raw .jsonl files (gitignored)
│   └── processed/                # products.jsonl, FAISS index, BM25 index
│
├── notebooks/
│   └── amazon_product_assistant.ipynb
│
├── src/
│   ├── preprocessor.py           # data cleaning + merging
│   ├── utils.py                  # tokenization + corpus building
│   ├── bm25.py                   # BM25 index + search
│   ├── semantic.py               # embeddings + FAISS index + search
│   ├── hybrid.py                 # RRF hybrid search
│   ├── tools.py                  # price + rating filters
│   ├── rag.py                    # RAG pipeline
│   ├── guardrails.py             # input + output validation
│   ├── retrieval_metrics.py      # Precision@K, MRR, NDCG
│   └── ragas_eval.py             # RAGAS evaluation
│
├── app/
│   └── app.py                    # Streamlit app
│
├── results/
│   ├── discussion.md             
│   └── ragas_results.json        
│
├── tests/
│   └── test_retrieval.py         
│
└── .github/
    └── workflows/
        └── ci.yml                
```

---

## 🚀 Setup

### 1. Clone the repository

```bash
git clone https://github.com/komalpreet10/Amazon_Product_Query_Assistant.git
cd Amazon_Product_Query_Assistant
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
```

### 5. Build indexes

```python
from src.preprocessor import build_products
from src.utils import build_corpus
from src.bm25 import build_bm25
from src.semantic import build_semantic_index

corpus, tokenized_corpus = build_corpus(products)
bm25 = build_bm25(tokenized_corpus)
index, embeddings = build_semantic_index(corpus)
```

---

## 🖥️ Run the App

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Run with Docker

```bash
docker build -t amazon-assistant .
docker run -p 8501:8501 --env-file .env amazon-assistant
```

---

## 🧪 Run Tests

```bash
pytest tests/ -v
```

---

## 📦 Dataset

- **Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — McAuley Lab, UCSD
- **Category:** All Beauty
- **Products:** 112,590
- **Reviews:** 701,528

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Data | HuggingFace Datasets, Amazon Reviews 2023 |
| Keyword Search | rank-bm25 |
| Semantic Search | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| LLM | OpenAI GPT-4o-mini |
| RAG Evaluation | RAGAS |
| App | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Language | Python 3.11 |
