# LoTTE RAG Search Engine

Hybrid Information Retrieval project (Project 21) — TF-IDF, BM25, Sentence Transformer and RRF,
with a React frontend, a FastAPI backend and an LLM answer layer.

## Setup

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd lotte-rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in your environment variables
cp .env.example .env
# Edit .env — set LLM_PROVIDER and LLM_API_KEY

# 5. Download NLTK data (run once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt_tab')"
```

> **Note:** The dense-embedding step (`scripts/embed.py`) must be run on a GPU (e.g. Google Colab free T4).
> All other steps run on a laptop.
