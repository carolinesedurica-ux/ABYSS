"""
Dark Data Miner — CLI entry point.

Commands:
  run    '<natural language>'          Full agentic pipeline (auto-detects intent)
  ingest  <file1> [file2 ...]          Ingest specific files directly
  query   '<question>'                 Query the knowledge base directly
  serve                                Start FastAPI server on :8000

Examples:
  python orchestrator.py run "What did the lead engineer say about supply chains?"
  python orchestrator.py run "Add the file meeting_2026_04.mp4"
  python orchestrator.py run "What files are indexed?"
  python orchestrator.py ingest meeting.mp4 manual.pdf
  python orchestrator.py query "What are the Q3 production targets?"
  python orchestrator.py serve
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "omno")


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_run(user_input: str, file_paths: list[str] | None = None) -> None:
    """Full agentic pipeline — Flash routes, Pro synthesizes."""
    from dark_data_miner.agents.router_agent import build_pipeline

    print(f'\nRunning agentic pipeline for: "{user_input}"\n')
    pipeline = build_pipeline()
    result = pipeline.invoke({
        "user_input": user_input,
        "intent": "query",
        "file_paths": file_paths or [],
        "answer": "",
        "citations": [],
        "processed": [],
        "total_chunks": 0,
        "errors": [],
    })

    print(f"\n{result['answer']}\n")
    if result["citations"]:
        print("Sources:")
        for c in result["citations"]:
            print(f"  [{c['index']}] {c['label']}  (score: {c['score']})")
    if result.get("errors"):
        print("Errors:")
        for e in result["errors"]:
            print(f"  {e}")


def cmd_ingest(paths: list[str]) -> None:
    from dark_data_miner.agents.ingestion_agent import build_ingestion_graph

    print(f"Ingesting {len(paths)} file(s)...")
    graph = build_ingestion_graph()
    result = graph.invoke({"file_paths": paths, "processed": [], "errors": [], "total_chunks": 0})

    print(f"\nDone — {len(result['processed'])} file(s) ingested, {result['total_chunks']} chunks stored.")
    if result["errors"]:
        print("Errors:")
        for e in result["errors"]:
            print(f"  {e}")


def cmd_query(question: str) -> None:
    from dark_data_miner.agents.query_agent import build_query_graph

    print(f'Searching for: "{question}"\n')
    graph = build_query_graph()
    result = graph.invoke({
        "original_query": question,
        "query": question,
        "chunks": [],
        "scores": [],
        "retrieval_sufficient": False,
        "retry_count": 0,
        "answer": "",
        "citations": [],
    })

    print(f"Answer:\n\n{result['answer']}\n")
    if result["citations"]:
        print("Sources:")
        for c in result["citations"]:
            print(f"  [{c['index']}] {c['label']}  (score: {c['score']})")


def cmd_serve() -> None:
    import uvicorn
    uvicorn.run("dark_data_miner.api.server:app", host="0.0.0.0", port=8000, reload=True)


# ── CLI dispatch ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        if len(sys.argv) < 3:
            print("Usage: python orchestrator.py run '<natural language input>'")
            sys.exit(1)
        cmd_run(sys.argv[2])

    elif command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python orchestrator.py ingest <file1> [file2 ...]")
            sys.exit(1)
        cmd_ingest(sys.argv[2:])

    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python orchestrator.py query '<question>'")
            sys.exit(1)
        cmd_query(sys.argv[2])

    elif command == "serve":
        cmd_serve()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)
