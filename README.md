🛍️ Amazon Product Query Assistant
An end-to-end Retrieval-Augmented Generation (RAG) system built over 112K Amazon Beauty products and 700K reviews. The system supports hybrid retrieval (BM25 + Semantic Search), tool-augmented RAG with GPT-4o-mini, input/output guardrails, and a Streamlit web app.

🏗️ Architecture
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

✨ Features

Hybrid Retrieval — combines BM25 keyword search and semantic search (all-MiniLM-L6-v2 + FAISS) using Reciprocal Rank Fusion (RRF)
RAG Pipeline — GPT-4o-mini generates answers grounded in retrieved product context
Tool Integration — automatic price and rating filter detection from natural language queries
Guardrails — input validation (off-topic, harmful, gibberish detection) and output validation (medical claims, grounding checks)
Evaluation — retrieval quality (Precision@5, MRR, NDCG@5) and RAG quality (RAGAS faithfulness, answer relevancy)
Streamlit App — interactive web app with BM25 / Semantic / Hybrid / RAG search modes
Docker + CI/CD — containerized with Docker, automated testing with GitHub Actions


📊 Evaluation Results
Retrieval Metrics
MethodPrecision@5MRRNDCG@5BM250.600.750.60Semantic0.660.830.69Hybrid0.780.730.71
RAGAS Metrics
MetricScoreFaithfulness0.57Answer Relevancy0.60

📁 Project Structure
Amazon_Product_Query_Assistant/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .env                          # never commit secrets
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
│   ├── discussion.md             # evaluation discussion
│   └── ragas_results.json        # RAGAS scores
│
├── tests/
│   └── test_retrieval.py         # unit tests
│
└── .github/
    └── workflows/
        └── ci.yml                # GitHub Actions CI/CD

🚀 Setup
1. Clone the repository
bashgit clone https://github.com/your-username/Amazon_Product_Query_Assistant.git
cd Amazon_Product_Query_Assistant
2. Create virtual environment
bashpython -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Set up environment variables
bashcp .env.example .env
# edit .env and add your API keys
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxx
5. Download and process data
Run the notebook or:
pythonfrom src.preprocessor import build_products
# loads from HuggingFace and saves to data/processed/products.jsonl
6. Build indexes
pythonfrom src.bm25 import build_bm25
from src.semantic import build_semantic_index
from src.utils import build_corpus

corpus, tokenized_corpus = build_corpus(products)
bm25 = build_bm25(tokenized_corpus)
index, embeddings = build_semantic_index(corpus)

🖥️ Run the App
bashstreamlit run app/app.py
Open http://localhost:8501 in your browser.

🐳 Run with Docker
bashdocker build -t amazon-assistant .
docker run -p 8501:8501 --env-file .env amazon-assistant

🧪 Run Tests
bashpytest tests/ -v

📦 Dataset

Source: Amazon Reviews 2023 — McAuley Lab, UCSD
Category: All Beauty
Products: 112,590
Reviews: 701,528
🛠️ Tech Stack
ComponentTechnologyDataHuggingFace Datasets, Amazon Reviews 2023Keyword Searchrank-bm25Semantic Searchsentence-transformers (all-MiniLM-L6-v2)Vector StoreFAISSLLMOpenAI GPT-4o-miniRAG EvaluationRAGASAppStreamlitContainerizationDockerCI/CDGitHub ActionsLanguagePython 3.11
