"""Build the Chroma vector index from everything in docs/.

Run once after adding or changing documents:

    python ingest.py
"""

import shutil
import sys

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
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
}


def load_documents():
    """Load every supported file in docs/ into LangChain Document objects."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")

    documents = []
    for path in sorted(DOCS_DIR.iterdir()):
        loader_cls = LOADERS.get(path.suffix.lower())
        if loader_cls is None:
            print(f"  skipping unsupported file: {path.name}")
            continue
        kwargs = {"encoding": "utf-8"} if loader_cls is TextLoader else {}
        loaded = loader_cls(str(path), **kwargs).load()
        print(f"  loaded {path.name} ({len(loaded)} page(s))")
        documents.extend(loaded)
    return documents


def main():
    require_google_key()

    if CHROMA_DIR.exists():
        print(f"Removing stale index at {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f"Loading documents from {DOCS_DIR}")
    documents = load_documents()
    if not documents:
        print("No documents found. Add .md, .txt or .pdf files to docs/ first.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, google_api_key=require_google_key()
    )
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="project_docs",
    )
    print(f"Indexed {vectorstore._collection.count()} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
