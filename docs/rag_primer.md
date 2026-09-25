# SAMPLE DOC 2 — replace with your own notes

# RAG Design Primer (my working notes)

## When to use RAG vs fine-tuning

Prefer **RAG** when the knowledge changes often, must stay up to date, or needs
per-user/per-tenant scoping and citations. Prefer **fine-tuning** when you want
to change the model's *behavior* (tone, output format, domain style) rather than
give it *facts*. Rule of thumb from my notes: facts → retrieval; style/behavior
→ fine-tuning; if you need both, do RAG first and fine-tune later.

## Chunking strategy

- Default: recursive character splitting at ~1000 chars with ~150 chars overlap.
  The overlap keeps sentences that straddle chunk boundaries retrievable.
- Markdown: split on heading boundaries first (keeps sections coherent), fall
  back to character splitting for oversized sections.
- Don't go below ~300 chars per chunk — chunks too small lose context and the
  grader/LLM sees fragments instead of facts.

## Embedding notes

- text-embedding-004 (Gemini) output dimension: 768 by default.
- Cosine similarity is the default metric in Chroma for these vectors.
- Re-index everything if you switch embedding models: vectors from different
  models are not comparable.

## Grading retrieved context

A cheap "grade then answer" step (ask the LLM: "does this context answer the
question?") catches most cases where top-k retrieval returns *related but
insufficient* chunks, and lets the agent fall back to another tool instead of
hallucinating from weak context.
