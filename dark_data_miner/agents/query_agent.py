"""
Agentic query pipeline with self-correcting retrieval.

                ┌─────────┐
    START ──▶   │ retrieve │
                └────┬─────┘
                     │
                ┌────▼─────┐
                │ evaluate │  ◀── Gemini Flash: are chunks relevant?
                └────┬─────┘
                     │
          ┌──────────┴──────────┐
     yes / retry≥2         no + retry<2
          │                     │
     ┌────▼──────┐        ┌─────▼──────┐
     │ synthesize│        │   refine   │  ◀── Flash rewrites the search query
     │ (Pro)     │        └─────┬──────┘
     └────┬──────┘              └──── loops back to retrieve
          │
         END
"""
import os
from typing import TypedDict, Literal

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from ..ingestion.embedder import search
from ..ingestion.models import DocumentChunk

MAX_RETRIES = 2

_SYNTHESIS_SYSTEM = (
    "You are an enterprise knowledge assistant. Answer questions using only "
    "the numbered source excerpts provided. Cite every claim with [N] notation. "
    "For video/audio sources include the timestamp (e.g. [2] meeting.mp4 @ 14:32). "
    "For PDF sources include the page number (e.g. [3] manual.pdf p.47). "
    "If the excerpts do not contain enough information to answer, say so explicitly — "
    "do not hallucinate or draw on outside knowledge."
)


class QueryState(TypedDict):
    original_query: str       # user's question — never mutated
    query: str                # current vector-search query (may be refined)
    chunks: list[DocumentChunk]
    scores: list[float]
    retrieval_sufficient: bool
    retry_count: int
    answer: str
    citations: list[dict]


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


# ── nodes ─────────────────────────────────────────────────────────────────────

def _retrieve(state: QueryState) -> QueryState:
    hits = search(state["query"], n_results=8)
    return {**state, "chunks": [h[0] for h in hits], "scores": [h[1] for h in hits]}


def _evaluate(state: QueryState) -> QueryState:
    """Gemini 1.5 Flash: are retrieved chunks relevant to the original question?"""
    if not state["chunks"]:
        return {**state, "retrieval_sufficient": False}

    sample = "\n---\n".join(c.content[:300] for c in state["chunks"][:4])
    client = _client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f"Question: {state['original_query']}\n\n"
            f"Retrieved excerpts (sample):\n{sample}\n\n"
            "Do these excerpts contain information relevant enough to answer the question? "
            "Reply with exactly one word: 'yes' or 'no'."
        ),
    )
    sufficient = resp.text.strip().lower().startswith("yes")
    print(f"    [evaluate] sufficient={sufficient}  retry={state['retry_count']}")
    return {**state, "retrieval_sufficient": sufficient}


def _refine(state: QueryState) -> QueryState:
    """Gemini 1.5 Flash: rewrite the search query to find better results."""
    client = _client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            f'Original question: "{state["original_query"]}"\n'
            f'Search query used: "{state["query"]}"\n'
            "The results were not relevant. Write a better search query using different "
            "vocabulary or a more specific angle. Reply with only the new query — "
            "no explanation, no quotes."
        ),
    )
    new_query = resp.text.strip()
    print(f"    [refine]   new query: {new_query!r}")
    return {**state, "query": new_query, "retry_count": state["retry_count"] + 1}


def _synthesize(state: QueryState) -> QueryState:
    """Gemini 1.5 Pro: synthesize final answer from retrieved chunks."""
    chunks = state["chunks"]
    if not chunks:
        return {**state, "answer": "No relevant content found in the knowledge base.", "citations": []}

    context = "\n\n".join(
        f"[{i}] {c.citation_label()}\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )
    client = _client()
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(
            system_instruction=_SYNTHESIS_SYSTEM,
            max_output_tokens=2048,
            temperature=0.1,
        ),
        contents=f"Source excerpts:\n\n{context}\n\nQuestion: {state['original_query']}",
    )
    citations = [
        {
            "index": i + 1,
            "label": c.citation_label(),
            "source_file": c.source_file,
            "source_type": c.source_type,
            "timestamp_start": c.timestamp_start,
            "page_number": c.page_number,
            "score": round(state["scores"][i], 4),
        }
        for i, c in enumerate(chunks)
    ]
    return {**state, "answer": response.text, "citations": citations}


# ── routing ───────────────────────────────────────────────────────────────────

def _after_evaluate(state: QueryState) -> Literal["synthesize", "refine"]:
    if state["retrieval_sufficient"] or state["retry_count"] >= MAX_RETRIES:
        return "synthesize"
    return "refine"


# ── graph ─────────────────────────────────────────────────────────────────────

def build_query_graph():
    g = StateGraph(QueryState)
    g.add_node("retrieve", _retrieve)
    g.add_node("evaluate", _evaluate)
    g.add_node("refine", _refine)
    g.add_node("synthesize", _synthesize)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_conditional_edges("evaluate", _after_evaluate, {
        "synthesize": "synthesize",
        "refine": "refine",
    })
    g.add_edge("refine", "retrieve")
    g.add_edge("synthesize", END)

    return g.compile()
