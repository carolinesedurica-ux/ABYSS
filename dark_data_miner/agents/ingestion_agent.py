"""
Ingestion pipeline with Gemini-powered routing:

  Gemini 1.5 Flash  → classifies PDF pages (text vs. scanned) to pick the right extractor
  Gemini 1.5 Pro    → native video/audio understanding (visuals + audio, not just speech)
  faster-whisper    → local fallback when USE_GEMINI_VISION=false

File routing decision tree:
  PDF  →  Flash checks first page → text: PyMuPDF | scanned: Gemini Flash OCR per page
  video/audio  →  USE_GEMINI_VISION=true: Gemini Pro | false: local Whisper
"""
import os
from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF — used to render first page for Flash classification
from langgraph.graph import StateGraph, END

from ..ingestion.embedder import embed_and_store
from ..ingestion.gemini_ingestor import (
    VIDEO_EXTS,
    AUDIO_EXTS,
    classify_pdf_with_flash,
    ingest_video_gemini,
)
from ..ingestion.pdf_ingestor import ingest_pdf
from ..ingestion.video_ingestor import ingest_video
from ..ingestion.models import DocumentChunk


class IngestionState(TypedDict):
    file_paths: list[str]
    processed: list[str]
    errors: list[str]
    total_chunks: int


def _route_and_ingest(file_path: str) -> list[DocumentChunk]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in VIDEO_EXTS | AUDIO_EXTS:
        use_gemini = os.getenv("USE_GEMINI_VISION", "true").lower() == "true"
        if use_gemini:
            print(f"    → Gemini 1.5 Pro (native video understanding)")
            return ingest_video_gemini(file_path)
        else:
            print(f"    → faster-whisper (local)")
            return ingest_video(file_path)

    if suffix == ".pdf":
        doc = fitz.open(file_path)
        first_page_png = doc[0].get_pixmap(dpi=100).tobytes("png")
        doc.close()

        pdf_type = classify_pdf_with_flash(first_page_png)
        print(f"    → Flash classified PDF as '{pdf_type}'")

        if pdf_type == "text":
            return ingest_pdf(file_path)
        else:
            # Scanned PDF: use Gemini Flash to OCR each page
            return _ocr_pdf_with_flash(file_path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _ocr_pdf_with_flash(file_path: str) -> list[DocumentChunk]:
    """Render each PDF page as PNG and OCR it with Gemini 1.5 Flash."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    path = Path(file_path)
    doc = fitz.open(file_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        png_bytes = page.get_pixmap(dpi=150).tobytes("png")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                "Extract all text from this scanned PDF page. Output only the extracted text, no commentary.",
            ],
        )
        text = response.text.strip()
        if text:
            chunks.append(DocumentChunk(
                source_file=path.name,
                source_type="pdf",
                content=text,
                page_number=page_num,
            ))

    doc.close()
    return chunks


def _ingest_node(state: IngestionState) -> IngestionState:
    processed, errors, total_chunks = [], [], 0

    for file_path in state["file_paths"]:
        path = Path(file_path)
        try:
            print(f"  Processing {path.name}...")
            chunks = _route_and_ingest(file_path)
            embed_and_store(chunks)
            processed.append(str(path))
            total_chunks += len(chunks)
            print(f"  [ok] {path.name} — {len(chunks)} chunks")
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            print(f"  [err] {path.name}: {exc}")

    return {**state, "processed": processed, "errors": errors, "total_chunks": total_chunks}


def build_ingestion_graph():
    g = StateGraph(IngestionState)
    g.add_node("ingest", _ingest_node)
    g.set_entry_point("ingest")
    g.add_edge("ingest", END)
    return g.compile()
