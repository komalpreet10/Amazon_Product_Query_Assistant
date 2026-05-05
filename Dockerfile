# ── base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── set working directory ───────────────────────────────────────────────────
WORKDIR /app

# ── install system dependencies ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── copy requirements ─────────────────────────────
COPY requirements.txt .

# ── install Python dependencies ─────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── copy source code ────────────────────────────────────────────────────────
COPY src/ ./src/
COPY app/ ./app/
COPY data/processed/ ./data/processed/

# ── expose Streamlit port ───────────────────────────────────────────────────
EXPOSE 8501

# ── run app ─────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]