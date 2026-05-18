"""
Extracts text from PDF files using PyMuPDF (fitz).
Each page is chunked at CHUNK_SIZE characters, breaking at word boundaries.
Page number is carried through so citations can point back to exact pages.
"""
from pathlib import Path
import fitz
from .models import DocumentChunk

CHUNK_SIZE = 1000  # characters per chunk


def ingest_pdf(file_path: str) -> list[DocumentChunk]:
    path = Path(file_path)
    doc = fitz.open(file_path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue

        for sub in _chunk_text(text, CHUNK_SIZE):
            chunks.append(DocumentChunk(
                source_file=path.name,
                source_type="pdf",
                content=sub,
                page_number=page_num,
            ))

    doc.close()
    return chunks


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text at word boundaries, keeping chunks under max_chars."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    words = text.split()
    buf: list[str] = []
    buf_len = 0

    for word in words:
        word_len = len(word) + 1
        if buf_len + word_len > max_chars and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += word_len

    if buf:
        chunks.append(" ".join(buf))

    return chunks
