"""LangGraph orchestration: router -> retriever -> grader -> (fallback) -> generator.

Flow:
    START
      └─> route_question        (LLM picks: docs / web / direct)
            ├─ "docs"  ─> retrieve_docs ─> grade_docs ─┬─ useful        ─> generate
            │                                          └─ insufficient  ─> run_web_search ─> generate
            ├─ "web"   ─> run_web_search ─> generate
            └─ "direct" ─> generate
    generate ─> END

The grader is the fallback mechanism: if the retrieved chunks don't actually
answer the question, the graph routes to web search instead of letting the
LLM hallucinate from weak context.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from config import CHROMA_DIR, LLM_MODEL, require_google_key
from tools import search_docs, web_search


class AgentState(TypedDict):
    question: str
    route: str
    doc_context: str
    doc_verdict: str
    web_context: str
    source: Literal["docs", "web", "direct"]
    answer: str
    # operator.add makes every node's update append to the trace
    # (LangGraph's default for list state is overwrite)
    steps: Annotated[list[str], operator.add]


llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, google_api_key=require_google_key(), temperature=0
)


# --- structured decision outputs -------------------------------------------------

class Route(BaseModel):
    decision: Literal["docs", "web", "direct"] = Field(
        description="Where the answer should come from."
    )


class DocGrade(BaseModel):
    verdict: Literal["useful", "insufficient"] = Field(
        description="Whether the retrieved context can answer the question."
    )


ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """You are the router of a research agent. The user has a private collection
of documents (project notes, READMEs, personal docs).

Decide where the answer should come from:
- "docs": the question is about the user's own projects, notes, decisions, or
  could plausibly be answered by their documents.
- "web": the question needs public or current knowledge (news, general facts,
  library docs, prices, releases).
- "direct": greetings, small talk, or meta questions about this conversation.

Question: {question}"""
)

GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading retrieved documents for relevance.

Question: {question}

Retrieved context:
{context}

Does this context contain the information needed to answer the question?
Grade strictly: weak or partially related context counts as insufficient."""
)

GENERATE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful research assistant.

Answer the question using ONLY the context below.

Context ({source_label}):
{context}

Question: {question}

Guidelines:
- Be concise (3-6 sentences unless the question needs more).
- Reference specifics from the context.
- If the context is insufficient, say so plainly instead of guessing."""
)

router_chain = ROUTER_PROMPT | llm.with_structured_output(Route)
grade_chain = GRADE_PROMPT | llm.with_structured_output(DocGrade)
generate_chain = GENERATE_PROMPT | llm


# --- nodes -----------------------------------------------------------------------

def route_question(state: AgentState) -> dict:
    decision = router_chain.invoke({"question": state["question"]}).decision
    note = ""
    if decision == "docs" and not CHROMA_DIR.exists():
        # No index has been built yet (ingest.py never run) — a docs lookup
        # can only fail, so fall back to web instead of burning two LLM calls.
        decision, note = "web", " (no doc index — fell back to web)"
    return {"route": decision, "steps": [f"router -> {decision}{note}"]}


def retrieve_docs(state: AgentState) -> dict:
    context = search_docs.invoke({"query": state["question"]})
    return {"doc_context": context, "steps": ["retrieve_docs"]}


def grade_docs(state: AgentState) -> dict:
    verdict = grade_chain.invoke(
        {"question": state["question"], "context": state["doc_context"]}
    ).verdict
    return {"doc_verdict": verdict, "steps": [f"grader -> {verdict}"]}


def run_web_search(state: AgentState) -> dict:
    context = web_search.invoke({"query": state["question"]})
    return {"web_context": context, "steps": ["web_search"]}


def generate(state: AgentState) -> dict:
    if state.get("doc_verdict") == "useful":
        context, source = state["doc_context"], "docs"
    elif state.get("web_context"):
        context, source = state["web_context"], "web"
    elif state.get("doc_context"):
        context, source = state["doc_context"], "docs"
    else:
        context, source = "(no tools used)", "direct"

    source_label = {
        "docs": "the user's private documents",
        "web": "public web search results",
        "direct": "none — answer from general knowledge",
    }[source]

    answer = generate_chain.invoke(
        {
            "context": context,
            "question": state["question"],
            "source_label": source_label,
        }
    ).content
    return {"source": source, "answer": answer, "steps": [f"generate ({source})"]}


# --- conditional edges -----------------------------------------------------------

def after_router(state: AgentState) -> Literal["retrieve_docs", "run_web_search", "generate"]:
    return {
        "docs": "retrieve_docs",
        "web": "run_web_search",
        "direct": "generate",
    }[state["route"]]


def after_grade(state: AgentState) -> Literal["run_web_search", "generate"]:
    return "generate" if state["doc_verdict"] == "useful" else "run_web_search"


builder = StateGraph(AgentState)
builder.add_node("route_question", route_question)
builder.add_node("retrieve_docs", retrieve_docs)
builder.add_node("grade_docs", grade_docs)
builder.add_node("run_web_search", run_web_search)
builder.add_node("generate", generate)

builder.add_edge(START, "route_question")
builder.add_conditional_edges(
    "route_question",
    after_router,
    {
        "retrieve_docs": "retrieve_docs",
        "run_web_search": "run_web_search",
        "generate": "generate",
    },
)
builder.add_edge("retrieve_docs", "grade_docs")
builder.add_conditional_edges(
    "grade_docs",
    after_grade,
    {"run_web_search": "run_web_search", "generate": "generate"},
)
builder.add_edge("run_web_search", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
