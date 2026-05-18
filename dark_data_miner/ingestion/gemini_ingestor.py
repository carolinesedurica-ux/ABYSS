"""
Native video/audio ingestion via Gemini 1.5 Pro.

Unlike faster-whisper (audio-only), Gemini 1.5 Pro understands the full video:
whiteboards, slides, diagrams, on-screen text, and speaker gestures are all captured
in the transcript. Supports up to ~1 hour of video per file.

Flow: upload to Gemini File API → poll until ACTIVE → transcribe → delete remote file.
"""
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import types

from .models import DocumentChunk

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

MIME_MAP = {
    ".mp4": "video/mp4",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

TRANSCRIPT_PROMPT = """
Produce a detailed timestamped transcript of this recording.

Rules:
- Format every segment as: [HH:MM:SS] <spoken text or visual description>
- Transcribe all spoken words verbatim. Preserve technical terms, acronyms, and names.
- When a speaker references something visual (a slide, whiteboard, diagram, screen share),
  add a brief description in parentheses immediately after, e.g.:
  [00:04:12] "...as you can see in this chart (bar chart: Q3 revenue vs forecast, gap of 18%)..."
- One segment per natural pause or topic shift. Do not merge long passages.
- Do not summarize or paraphrase. Do not add commentary.
"""

_TS_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(.+)$")


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def _ts_to_seconds(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def ingest_video_gemini(file_path: str) -> list[DocumentChunk]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    mime_type = MIME_MAP.get(suffix, "video/mp4")
    source_type = "audio" if suffix in AUDIO_EXTS else "video"

    client = _client()

    # --- Upload ---
    print(f"  Uploading {path.name} to Gemini File API...")
    uploaded = client.files.upload(
        file=path,
        config={"mime_type": mime_type, "display_name": path.name},
    )

    # --- Poll until ACTIVE ---
    while uploaded.state.name == "PROCESSING":
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file processing failed: {uploaded.state.name}")

    # --- Transcribe with Gemini 1.5 Pro ---
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        config=types.GenerateContentConfig(temperature=0.0),
        contents=[
            types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime_type),
            TRANSCRIPT_PROMPT,
        ],
    )

    # --- Parse timestamped segments ---
    lines = response.text.strip().splitlines()
    parsed: list[tuple[float, str]] = []
    for line in lines:
        m = _TS_RE.match(line.strip())
        if m:
            h, mins, s, text = m.groups()
            parsed.append((_ts_to_seconds(h, mins, s), text.strip()))

    chunks: list[DocumentChunk] = []
    for i, (ts_start, text) in enumerate(parsed):
        ts_end = parsed[i + 1][0] if i + 1 < len(parsed) else ts_start + 30.0
        chunks.append(DocumentChunk(
            source_file=path.name,
            source_type=source_type,
            content=text,
            timestamp_start=ts_start,
            timestamp_end=ts_end,
        ))

    # --- Clean up remote file ---
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass  # non-fatal; files auto-expire in 48h

    return chunks


def classify_pdf_with_flash(first_page_png: bytes) -> str:
    """
    Use Gemini 1.5 Flash to determine if a PDF page is text-based or scanned.
    Returns 'text' or 'scanned'.
    """
    client = _client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=first_page_png, mime_type="image/png"),
            (
                "Look at this PDF page image. Is the text selectable/digital (rendered by a "
                "PDF engine), or is it a scanned photograph of a printed page? "
                "Reply with exactly one word: 'text' or 'scanned'."
            ),
        ],
    )
    answer = response.text.strip().lower()
    return "scanned" if "scanned" in answer else "text"
