# Doc-Aware Research Agent

A LangGraph agent that answers questions from **your own documents first**
(RAG), grades the retrieval, and **falls back to web search** when the docs
don't contain the answer. Every answer reports which source was used.

```
                     ┌─────────────────┐
        question ──> │  route_question │  LLM router: docs / web / direct
                     └───────┬─────────┘
            ┌────────────────┼──────────────────┐
         "docs"           "web"             "direct"
            │                │                  │
   ┌────────▼───────┐        │                  │
   │ retrieve_docs  │        │                  │
   │ (RAG over      │        │                  │
   │  Chroma)       │        │                  │
   └────────┬───────┘        │                  │
   ┌────────▼───────┐        │                  │
   │   grade_docs   │ LLM grades whether the chunks
   └───┬────────┬───┘ actually answer the question
       │        │
    useful   insufficient
       │        │
       │   ┌────▼──────────┐
       │   │ run_web_search│  Tavily (or DuckDuckGo, no key needed)
       │   └────┬──────────┘
       ▼        ▼
     ┌──────────────┐
     │   generate   │  cited answer + source label
     └──────────────┘
```

## Stack

| Piece | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | Explicit state machine for the decide → retrieve → grade → fallback loop |
| RAG chain + loading | LangChain | Document loaders, text splitter, prompt chains |
| LLM | Gemini `gemini-2.0-flash` | Fast + cheap; structured output for router/grader |
| Embeddings | Gemini `text-embedding-004` | Same API as the LLM, one key |
| Vector store | Chroma (local, persisted) | Free, no account, runs on disk |
| Web search | Tavily (free tier) with DuckDuckGo fallback | Works with or without an API key |
| Serving | CLI + Flask | `python app.py "..."` or `POST /ask` |

## Setup

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Gemini key
copy .env.example .env           # then edit .env
#    GOOGLE_API_KEY=...          (https://aistudio.google.com/apikey)
#    TAVILY_API_KEY=...          (optional — without it, DuckDuckGo is used)

# 4. Index your documents
python ingest.py                 # indexes everything in docs/
```

Add your own files (`.md`, `.txt`, `.pdf`) to `docs/` and re-run `python ingest.py`.
Two sample docs are included so the project runs out of the box.

## Usage

**CLI (one-shot):**

```bash
python app.py "What database does the Aurora dashboard use and why?"
python app.py "What's the latest LangGraph release?"
```

**CLI (interactive):**

```bash
python app.py
```

**Flask server:**

```bash
python app.py --serve
# then open http://localhost:5000, or:
curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" ^
     -d "{\"question\": \"When should I prefer RAG over fine-tuning?\"}"
```

Example response:

```json
{
  "answer": "Your notes say to prefer RAG when the knowledge changes often ...",
  "source": "docs",
  "sources": ["D:\\...\\docs\\rag_primer.md"],
  "steps": ["router -> docs", "retrieve_docs", "grader -> useful", "generate (docs)"]
}
```

## How the fallback works

1. A router LLM call classifies the question: **docs** / **web** / **direct**.
2. If routed to docs, a retriever fetches top-k chunks from Chroma.
3. A grader LLM call decides whether those chunks actually answer the question
   (`useful` / `insufficient`) — related-but-weak context counts as insufficient.
4. On `insufficient`, the graph routes to web search instead of answering from
   thin context, which is what prevents confident hallucinations.
5. The final answer carries a source label and a citation list (file paths for
   docs, URLs for web), plus a step-by-step trace of the route it took.
6. If no index has been built yet (`python ingest.py` never run), the router
   skips the docs route entirely and answers from the web.

## Project layout

```
doc-aware-research-agent/
├── app.py          # CLI + Flask entry points
├── graph.py        # LangGraph state machine (router, retriever, grader, generator)
├── tools.py        # search_docs (RAG) and web_search (Tavily/DuckDuckGo) tools
├── ingest.py       # loads docs/, chunks, embeds, writes Chroma index
├── config.py       # keys, model names, chunking parameters
├── docs/           # your documents (2 samples included)
├── chroma_db/      # persisted vector index (gitignored, created by ingest.py)
└── requirements.txt
```

## Notes & limits

- The index must be rebuilt (`python ingest.py`) after changing documents.
- Gemini free tier is rate-limited; the agent makes 2–4 LLM calls per question
  (router, optional grader, generator) plus 1 embedding call per retrieval.
- `chroma_db/` is local-only; delete it to reset the index.
- You may see a one-line `Direct use of automatic function calling ... is not
  recommended` warning from the Google SDK at startup. It's harmless noise from
  `langchain-google-genai` and doesn't affect behavior.
