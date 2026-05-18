from pydantic import BaseModel, Field
from typing import Optional, Literal
import uuid


class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str
    source_type: Literal["video", "audio", "pdf"]
    content: str

    # Video/audio — seconds from start of file
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None

    # PDF — 1-indexed
    page_number: Optional[int] = None

    def citation_label(self) -> str:
        if self.source_type in ("video", "audio") and self.timestamp_start is not None:
            return f"{self.source_file} @ {_fmt_ts(self.timestamp_start)}"
        if self.page_number is not None:
            return f"{self.source_file} p.{self.page_number}"
        return self.source_file

    def to_chroma_metadata(self) -> dict:
        return {
            "source_file": self.source_file,
            "source_type": self.source_type,
            "timestamp_start": self.timestamp_start if self.timestamp_start is not None else -1.0,
            "timestamp_end": self.timestamp_end if self.timestamp_end is not None else -1.0,
            "page_number": self.page_number if self.page_number is not None else -1,
        }

    @staticmethod
    def from_chroma_hit(doc: str, meta: dict, score: float) -> "DocumentChunk":
        return DocumentChunk(
            source_file=meta["source_file"],
            source_type=meta["source_type"],
            content=doc,
            timestamp_start=meta["timestamp_start"] if meta["timestamp_start"] != -1.0 else None,
            timestamp_end=meta["timestamp_end"] if meta["timestamp_end"] != -1.0 else None,
            page_number=meta["page_number"] if meta["page_number"] != -1 else None,
        )


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
