"""Build and update the Chroma vector index.

Two ways to use it:

  CLI (full rebuild from docs/):      python ingest.py
  Library (from app.py uploads):      ingest_paths([...]) -> chunk count

ingest_paths() is incremental: uploading a file replaces that file's chunks
and leaves the rest of the index untouched. A .docx upload needs the
optional `docx2txt` package (listed in requirements.txt).
"""

import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    require_google_key,
)

LOADERS = {
    ".md": TextLoader,
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
}
SUPPORTED = tuple(sorted(LOADERS))

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def _clean(documents):
    """Keep only the source in metadata - Chroma rejects complex values that
    PDF/DOCX loaders sometimes attach to documents."""
    for doc in documents:
        doc.metadata = {"source": doc.metadata.get("source", "")}
    return documents


def load_paths(paths):
    """Load files into LangChain Documents. Raises ValueError on unsupported types."""
    documents = []
    for p in paths:
        p = Path(p).resolve()
        loader_cls = LOADERS.get(p.suffix.lower())
        if loader_cls is None:
            raise ValueError(
                f"Unsupported file type '{p.suffix or '(none)'}' for {p.name}. "
                f"Supported: {', '.join(SUPPORTED)}"
            )
        kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        documents.extend(loader_cls(str(p), **kwargs).load())
    return _clean(documents)


def ingest_paths(paths, rebuild=False):
    """Ingest the given files into the Chroma index. Returns the chunk count.

    rebuild=True wipes the whole index first (used by the CLI full rebuild).
    Otherwise a re-uploaded file replaces its own previous chunks and the
    rest of the index is left untouched.
    """
    require_google_key()
    resolved = [Path(p).resolve() for p in paths]
    sources = [str(p) for p in resolved]

    documents = load_paths(resolved)
    chunks = _splitter.split_documents(documents)
    if not chunks:
        return 0

    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=require_google_key()
    )
    store = Chroma(
        collection_name="project_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    if not rebuild:
        # replace this file's previous chunks, keep everything else
        old = store.get(where={"source": {"$in": sources}})
        if old["ids"]:
            store.delete(ids=old["ids"])
    store.add_documents(chunks)
    return len(chunks)


def remove_source(path):
    """Delete one file's chunks from the index. Returns the number removed."""
    source = str(Path(path).resolve())
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=require_google_key()
    )
    store = Chroma(
        collection_name="project_docs",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    old = store.get(where={"source": source})
    if old["ids"]:
        store.delete(ids=old["ids"])
    return len(old["ids"])


def main():
    if not DOCS_DIR.exists():
        print(f"Docs directory not found: {DOCS_DIR}")
        sys.exit(1)
    paths = [
        p
        for p in sorted(DOCS_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in LOADERS
    ]
    if not paths:
        print(
            f"No supported documents in {DOCS_DIR} "
            f"(supported: {', '.join(SUPPORTED)})"
        )
        sys.exit(1)

    print(f"Indexing {len(paths)} file(s) from {DOCS_DIR}")
    n = ingest_paths(paths, rebuild=True)
    print(f"Done - {n} chunks indexed into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
