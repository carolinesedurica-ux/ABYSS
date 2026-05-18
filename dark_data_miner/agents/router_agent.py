"""
Top-level agentic pipeline.

Gemini 1.5 Flash classifies the user's natural language input, then routes
to the appropriate sub-pipeline:

    user_input
        │
   ┌────▼─────┐
   │  router  │  ◀── Gemini Flash: query | ingest | list
   └────┬─────┘
        │
   ┌────┴──────────────────┐
   │                       │                      │
   ▼                       ▼                      ▼
 query                  ingest               list_sources
(query_agent            (ingestion_agent     (ChromaDB metadata
 retrieve→evaluate       route+embed)         scan)
 →refine loop
 →Pro synthesis)
        │                  │                      │
        └──────────────────┴──────────────────────┘
                                │
                              answer + citations → END
"""
import os
from typing import TypedDict, Literal

import chromadb
from google import genai
from langgraph.graph import StateGraph, END

from .ingestion_agent import IngestionState, build_ingestion_graph
from .query_agent import QueryState, build_query_graph


class PipelineState(TypedDict):
    user_input: str
    intent: Literal["query", "ingest", "list"]
    file_paths: list[str]        # caller populates when intent is "ingest"
    answer: str
    citations: list[dict]
    processed: list[str]
    total_chunks: int
    errors: list[str]


# ── nodes ─────────────────────────────────────────────────────────────────────

def _router_node(state: PipelineState) -> PipelineState:
    """Gemini Flash classifies intent from natural language."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f'User input: "{state["user_input"]}"\n\n'
            "Classify this into exactly one intent:\n"
            "  query   — user is asking a question about documents already indexed\n"
            "  ingest  — user wants to add or process new files into the knowledge base\n"
            "  list    — user wants to see which files/sources are already indexed\n"
            "Reply with exactly one word: query, ingest, or list."
        ),
    )
    raw = resp.text.strip().lower()
    intent: Literal["query", "ingest", "list"] = raw if raw in {"query", "ingest", "list"} else "query"
    print(f"  [router] intent={intent!r}")
    return {**state, "intent": intent}


def _query_node(state: PipelineState) -> PipelineState:
    graph = build_query_graph()
    result: QueryState = graph.invoke({
        "original_query": state["user_input"],
        "query": state["user_input"],
        "chunks": [],
        "scores": [],
        "retrieval_sufficient": False,
        "retry_count": 0,
        "answer": "",
        "citations": [],
    })
    return {**state, "answer": result["answer"], "citations": result["citations"]}


def _ingest_node(state: PipelineState) -> PipelineState:
    graph = build_ingestion_graph()
    result: IngestionState = graph.invoke({
        "file_paths": state["file_paths"],
        "processed": [],
        "errors": [],
        "total_chunks": 0,
    })
    return {
        **state,
        "processed": result["processed"],
        "total_chunks": result["total_chunks"],
        "errors": result["errors"],
        "answer": (
            f"Ingested {len(result['processed'])} file(s), "
            f"{result['total_chunks']} chunks stored."
        ),
    }


def _list_sources_node(state: PipelineState) -> PipelineState:
    chroma_path = os.getenv("CHROMA_PATH", "./data/chroma_db")
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        col = client.get_collection("dark_data")
        metas = col.get(include=["metadatas"])["metadatas"]
        files = sorted({m["source_file"] for m in metas})
        if files:
            lines = "\n".join(f"  • {f}" for f in files)
            answer = f"Indexed sources ({len(files)} files):\n{lines}"
        else:
            answer = "No sources indexed yet."
    except Exception:
        answer = "No sources indexed yet."
    return {**state, "answer": answer, "citations": []}


# ── routing ───────────────────────────────────────────────────────────────────

def _route_intent(state: PipelineState) -> Literal["query", "ingest", "list"]:
    return state["intent"]


# ── graph ─────────────────────────────────────────────────────────────────────

def build_pipeline():
    g = StateGraph(PipelineState)
    g.add_node("router", _router_node)
    g.add_node("query", _query_node)
    g.add_node("ingest", _ingest_node)
    g.add_node("list", _list_sources_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", _route_intent, {
        "query": "query",
        "ingest": "ingest",
        "list": "list",
    })
    g.add_edge("query", END)
    g.add_edge("ingest", END)
    g.add_edge("list", END)

    return g.compile()
