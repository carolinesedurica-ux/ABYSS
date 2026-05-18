"""
Vault storage layer — raw file store + SQLite metadata.

Every file uploaded to the vault is:
  1. Copied to  VAULT_PATH/files/<uuid><ext>
  2. Recorded in VAULT_PATH/vault.db

Status lifecycle:  stored → mining → mined | error
"""
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _vault() -> Path:
    return Path(os.getenv("VAULT_PATH", "./data/vault"))


def _files_dir() -> Path:
    d = _vault() / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _db() -> sqlite3.Connection:
    db_path = _vault() / "vault.db"
    _vault().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vault_files (
            id           TEXT PRIMARY KEY,
            original_name TEXT NOT NULL,
            stored_name  TEXT NOT NULL,
            file_size    INTEGER DEFAULT 0,
            file_type    TEXT NOT NULL,
            uploaded_at  TEXT NOT NULL,
            mined_at     TEXT,
            chunk_count  INTEGER DEFAULT 0,
            status       TEXT DEFAULT 'stored'
        )
    """)
    conn.commit()
    return conn


def _file_type(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in VIDEO_EXTS: return "video"
    if ext in AUDIO_EXTS: return "audio"
    return "pdf"


def save_file(tmp_path: Path, original_name: str) -> dict:
    suffix = Path(original_name).suffix.lower()
    file_id = str(uuid.uuid4())
    stored_name = f"{file_id}{suffix}"
    dest = _files_dir() / stored_name
    shutil.copy2(str(tmp_path), str(dest))

    row = {
        "id": file_id,
        "original_name": original_name,
        "stored_name": stored_name,
        "file_size": dest.stat().st_size,
        "file_type": _file_type(original_name),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "mined_at": None,
        "chunk_count": 0,
        "status": "stored",
    }
    conn = _db()
    conn.execute("""
        INSERT INTO vault_files
        (id, original_name, stored_name, file_size, file_type, uploaded_at, mined_at, chunk_count, status)
        VALUES (:id, :original_name, :stored_name, :file_size, :file_type, :uploaded_at, :mined_at, :chunk_count, :status)
    """, row)
    conn.commit()
    conn.close()
    return row


def list_files() -> list[dict]:
    conn = _db()
    rows = conn.execute(
        "SELECT * FROM vault_files ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_file(file_id: str) -> Optional[dict]:
    conn = _db()
    row = conn.execute(
        "SELECT * FROM vault_files WHERE id = ?", (file_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_file_path(file_id: str) -> Optional[Path]:
    row = get_file(file_id)
    return (_files_dir() / row["stored_name"]) if row else None


def set_status(file_id: str, status: str,
               chunk_count: int = 0, mined_at: Optional[str] = None) -> None:
    conn = _db()
    conn.execute(
        "UPDATE vault_files SET status=?, chunk_count=?, mined_at=? WHERE id=?",
        (status, chunk_count, mined_at, file_id),
    )
    conn.commit()
    conn.close()


def delete_file(file_id: str) -> bool:
    row = get_file(file_id)
    if not row:
        return False
    path = _files_dir() / row["stored_name"]
    if path.exists():
        path.unlink()
    conn = _db()
    conn.execute("DELETE FROM vault_files WHERE id=?", (file_id,))
    conn.commit()
    conn.close()
    return True
