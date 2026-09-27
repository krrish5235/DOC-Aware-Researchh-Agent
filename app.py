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
import time
from pathlib import Path

from config import DOCS_DIR, GOOGLE_API_KEY

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

    @app.post("/upload")
    def upload_endpoint():
        """Accept one or more files, save them into docs/ and index them."""
        from ingest import SUPPORTED, ingest_paths
        from werkzeug.utils import secure_filename

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files received"}), 400

        DOCS_DIR.mkdir(exist_ok=True)
        saved, skipped = [], []
        for f in files:
            name = secure_filename(f.filename or "")
            if not name:
                skipped.append({"file": f.filename, "reason": "invalid file name"})
                continue
            ext = Path(name).suffix.lower()
            if ext not in SUPPORTED:
                skipped.append(
                    {
                        "file": f.filename,
                        "reason": f"unsupported type '{ext or '(none)'}' "
                        f"- supported: {', '.join(SUPPORTED)}",
                    }
                )
                continue
            dest = DOCS_DIR / name
            f.save(dest)  # same name = replace the file; ingest swaps its old chunks
            saved.append(dest)

        if not saved:
            return jsonify({"error": "No supported files", "skipped": skipped}), 400

        try:
            chunks = ingest_paths(saved)
        except Exception as exc:
            return (
                jsonify(
                    {
                        "error": f"Saved but indexing failed: {exc}",
                        "saved": [p.name for p in saved],
                    }
                ),
                500,
            )

        return jsonify(
            {
                "indexed": True,
                "chunks": chunks,
                "files": [p.name for p in saved],
                "skipped": skipped,
            }
        )

    @app.get("/api/files")
    def files_endpoint():
        """List the documents the agent can answer from."""
        from ingest import SUPPORTED

        items = []
        if DOCS_DIR.exists():
            for p in sorted(DOCS_DIR.iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED:
                    items.append(
                        {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
                    )
        return jsonify({"files": items})

    @app.delete("/api/files/<path:name>")
    def delete_file_endpoint(name):
        """Remove a document and its indexed chunks."""
        from ingest import remove_source

        dest = DOCS_DIR / Path(name).name
        if not dest.is_file():
            return jsonify({"error": "not found"}), 404
        removed = 0
        try:
            removed = remove_source(dest)
        except Exception:
            pass  # index may not exist yet - still allow removing the file
        dest.unlink()
        return jsonify({"deleted": dest.name, "chunks_removed": removed})

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
