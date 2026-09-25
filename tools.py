"""The agent's two tools: private document search (RAG) and public web search.

Both return plain strings formatted for an LLM context window.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    TOP_K,
    TAVILY_API_KEY,
    require_google_key,
)

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Lazily open the persisted Chroma index (created by ingest.py)."""
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"No vector index at {CHROMA_DIR}. Run `python ingest.py` first."
            )
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=require_google_key()
        )
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name="project_docs",
        )
    return _vectorstore


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


@tool
def search_docs(query: str) -> str:
    """Search the user's private document collection (project notes, READMEs,
    personal docs). Use for questions about the user's own projects, decisions,
    or anything that might live in their local documents."""
    try:
        docs = get_vectorstore().similarity_search(query, k=TOP_K)
    except RuntimeError as exc:
        return str(exc)
    if not docs:
        return "No matching documents found."
    return _format_docs(docs)


def _web_search_tavily(query: str) -> str | None:
    if not TAVILY_API_KEY:
        return None
    from tavily import TavilyClient

    client = TavilyClient(TAVILY_API_KEY)
    results = client.search(query, max_results=5)["results"]
    return _format_web_results(
        [{"title": r["title"], "url": r["url"], "body": r["content"]} for r in results]
    )


def _web_search_duckduckgo(query: str) -> str:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS

    raw = DDGS().text(query, max_results=5)
    return _format_web_results(
        [
            {"title": r.get("title", ""), "url": r.get("href", r.get("url", "")), "body": r.get("body", "")}
            for r in raw
        ]
    )


def _format_web_results(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(f"[{i}] {r['title']}\nurl: {r['url']}\n{r['body']}")
    return "\n\n".join(parts)


@tool
def web_search(query: str) -> str:
    """Search the public web. Use for current events, general knowledge,
    library/framework documentation, or anything not covered by the user's
    private documents."""
    try:
        result = _web_search_tavily(query)
        provider = "tavily"
        if result is None:
            result = _web_search_duckduckgo(query)
            provider = "duckduckgo"
    except Exception as exc:
        return f"Web search failed: {exc}"
    if not result:
        return "No web results found."
    return f"(via {provider})\n\n{result}"
