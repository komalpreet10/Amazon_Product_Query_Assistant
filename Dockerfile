# ── base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── set working directory ───────────────────────────────────────────────────
WORKDIR /app
ENV PYTHONUNBUFFERED=1

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
COPY api/ ./api/
RUN mkdir -p ./data/processed

# ── expose API port ─────────────────────────────────────────────────────────
EXPOSE 8000

# ── run API ─────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
