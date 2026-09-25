"""Central configuration: API keys, model names, paths, RAG parameters."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# RAG parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def require_google_key() -> str:
    """Return the Gemini API key or raise with a setup hint."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://aistudio.google.com/apikey"
        )
    return GOOGLE_API_KEY
