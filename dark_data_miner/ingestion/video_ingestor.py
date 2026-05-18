"""
Transcribes video and audio files using faster-whisper.
Each Whisper segment becomes one DocumentChunk with its timestamp preserved.
faster-whisper accepts raw MP4/MKV/MOV/MP3/WAV — ffmpeg handles demuxing internally.
"""
import os
from pathlib import Path
from faster_whisper import WhisperModel
from .models import DocumentChunk

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        model_size = os.getenv("WHISPER_MODEL", "base")
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _model


def ingest_video(file_path: str) -> list[DocumentChunk]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    source_type = "video" if suffix in VIDEO_EXTS else "audio"

    model = _get_model()
    segments, _ = model.transcribe(file_path, beam_size=5, word_timestamps=False)

    chunks = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        chunks.append(DocumentChunk(
            source_file=path.name,
            source_type=source_type,
            content=text,
            timestamp_start=seg.start,
            timestamp_end=seg.end,
        ))

    return chunks
