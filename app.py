"""Entry points: a CLI and a minimal Flask server.

CLI:
    python app.py "What chunking strategy do my notes recommend?"
    python app.py                      # interactive mode

Flask:
    python app.py --serve              # http://localhost:5000
    curl -X POST http://localhost:5000/ask -H "Content-Type: application/json" \
         -d '{"question": "What is LangGraph?"}'
"""

import argparse
import re
import sys

from config import GOOGLE_API_KEY

graph = None  # imported lazily in ask() so `--help` works without an API key


def _extract_doc_sources(doc_context: str) -> list[str]:
    return sorted({m for m in re.findall(r"source: (.+)", doc_context or "")})


def _extract_web_sources(web_context: str) -> list[str]:
    return sorted({m for m in re.findall(r"url: (\S+)", web_context or "")})


def ask(question: str) -> dict:
    """Run the graph and return {answer, source, sources, steps, elapsed}."""
    global graph
    if graph is None:
        from graph import graph as g

        graph = g
    import time

    t0 = time.perf_counter()
    result = graph.invoke({"question": question, "steps": []})

    if result["source"] == "docs":
        sources = _extract_doc_sources(result["doc_context"])
    elif result["source"] == "web":
        sources = _extract_web_sources(result["web_context"])
    else:
        sources = []

    return {
        "question": question,
        "answer": result["answer"],
        "source": result["source"],
        "sources": sources,
        "steps": result["steps"],
        "elapsed": round(time.perf_counter() - t0, 1),
    }


def print_answer(result: dict) -> None:
    print(f"\n[trace] {' -> '.join(result['steps'])}")
    print(f"[source] {result['source']}")
    print(f"\n{result['answer']}\n")
    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  - {s}")


def run_cli() -> None:
    print("Interactive mode — Ctrl+C to exit.")
    while True:
        try:
            question = input("\nyou > ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question:
            continue
        print_answer(ask(question))


def create_app():
    from flask import Flask, jsonify, render_template, request

    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/ask")
    def ask_endpoint():
        question = (request.get_json(silent=True) or {}).get("question", "").strip()
        if not question:
            return jsonify({"error": "Send {\"question\": \"...\"} as JSON"}), 400
        try:
            return jsonify(ask(question))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doc-Aware Research Agent")
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("--serve", action="store_true", help="start the Flask server")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print(
            "GOOGLE_API_KEY is not set.\n"
            "  1. copy .env.example .env\n"
            "  2. edit .env and paste your key from https://aistudio.google.com/apikey\n"
            "  3. python ingest.py   (once, to build the document index)\n"
            "Then run this command again."
        )
        sys.exit(1)

    if args.serve:
        create_app().run(host="127.0.0.1", port=args.port)
    elif args.question:
        print_answer(ask(" ".join(args.question)))
    else:
        run_cli()
