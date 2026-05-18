"""
Two-node LangGraph: retrieve → synthesize.

Synthesis uses Gemini 1.5 Pro — its 1M-token context window means the full
retrieved context fits in a single call without truncation, even for large
document sets.
"""
import os
from typing import TypedDict

from google import genai
from google.genai import types  # noqa: F401 — kept for parity; used in GenerateContentConfig
from langgraph.graph import StateGraph, END

from ..ingestion.embedder import search
from ..ingestion.models import DocumentChunk

SYSTEM_PROMPT = (
    "You are an enterprise knowledge assistant. Answer questions using only "
    "the numbered source excerpts provided. Cite every claim with [N] notation. "
    "For video/audio sources include the timestamp (e.g. [2] meeting.mp4 @ 14:32). "
    "For PDF sources include the page number (e.g. [3] manual.pdf p.47). "
    "If the excerpts do not contain enough information to answer, say so explicitly — "
    "do not hallucinate or draw on outside knowledge."
)


class QueryState(TypedDict):
    query: str
    chunks: list[DocumentChunk]
    scores: list[float]
    answer: str
    citations: list[dict]


def _get_model() -> genai.GenerativeModel:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    return genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            max_output_tokens=2048,
            temperature=0.1,
        ),
    )


def _retrieve_node(state: QueryState) -> QueryState:
    hits = search(state["query"], n_results=8)
    return {**state, "chunks": [h[0] for h in hits], "scores": [h[1] for h in hits]}


def _synthesize_node(state: QueryState) -> QueryState:
    chunks = state["chunks"]
    if not chunks:
        return {**state, "answer": "No relevant content found in the knowledge base.", "citations": []}

    context = "\n\n".join(
        f"[{i}] {chunk.citation_label()}\n{chunk.content}"
        for i, chunk in enumerate(chunks, start=1)
    )

    response = _get_model().generate_content(
        f"Source excerpts:\n\n{context}\n\nQuestion: {state['query']}"
    )

    citations = [
        {
            "index": i + 1,
            "label": chunk.citation_label(),
            "source_file": chunk.source_file,
            "source_type": chunk.source_type,
            "timestamp_start": chunk.timestamp_start,
            "page_number": chunk.page_number,
            "score": round(state["scores"][i], 4),
        }
        for i, chunk in enumerate(chunks)
    ]

    return {**state, "answer": response.text, "citations": citations}


def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("synthesize", _synthesize_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
