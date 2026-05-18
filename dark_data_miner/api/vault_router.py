"""
Vault API router — upload, list, delete, and mine raw files.

POST /vault/upload          — upload files into the vault (raw storage only)
GET  /vault/files           — list all vault files with metadata
DELETE /vault/files/{id}    — remove a file from the vault
POST /vault/mine/{id}       — trigger ingestion of one file (background task)
POST /vault/mine-all        — trigger ingestion of all unmined files
"""
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ..agents.ingestion_agent import build_ingestion_graph
from ..vault.store import (
    delete_file, get_file, get_file_path,
    list_files, save_file, set_status,
)

router = APIRouter(prefix="/vault", tags=["vault"])

SUPPORTED_EXTS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm",
    ".mp3", ".wav", ".m4a", ".ogg", ".flac",
    ".pdf",
}


@router.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    tmp = Path(tempfile.mkdtemp())
    saved = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise HTTPException(400, f"Unsupported type: {suffix}")
        tmp_path = tmp / upload.filename
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        row = save_file(tmp_path, upload.filename)
        saved.append(row)

    return {"uploaded": saved, "count": len(saved)}


@router.get("/files")
def get_files():
    files = list_files()
    stats = {
        "total":  len(files),
        "stored": sum(1 for f in files if f["status"] == "stored"),
        "mining": sum(1 for f in files if f["status"] == "mining"),
        "mined":  sum(1 for f in files if f["status"] == "mined"),
        "error":  sum(1 for f in files if f["status"] == "error"),
    }
    return {"files": files, "stats": stats}


@router.delete("/files/{file_id}")
def remove(file_id: str):
    if not delete_file(file_id):
        raise HTTPException(404, "File not found in vault")
    return {"deleted": file_id}


@router.post("/mine/{file_id}")
def mine_one(file_id: str, background_tasks: BackgroundTasks):
    row = get_file(file_id)
    if not row:
        raise HTTPException(404, "File not found in vault")
    if row["status"] == "mining":
        raise HTTPException(409, "Already mining this file")

    set_status(file_id, "mining")
    background_tasks.add_task(_mine_task, file_id)
    return {"status": "mining", "file_id": file_id, "name": row["original_name"]}


@router.post("/mine-all")
def mine_all(background_tasks: BackgroundTasks):
    files = [f for f in list_files() if f["status"] in ("stored", "error")]
    for f in files:
        set_status(f["id"], "mining")
        background_tasks.add_task(_mine_task, f["id"])
    return {"queued": len(files), "ids": [f["id"] for f in files]}


# ── background mining task ────────────────────────────────────

def _mine_task(file_id: str) -> None:
    path = get_file_path(file_id)
    if not path or not path.exists():
        set_status(file_id, "error")
        return
    try:
        graph = build_ingestion_graph()
        result = graph.invoke({
            "file_paths": [str(path)],
            "processed": [],
            "errors": [],
            "total_chunks": 0,
        })
        if result["errors"]:
            set_status(file_id, "error")
        else:
            set_status(
                file_id, "mined",
                chunk_count=result["total_chunks"],
                mined_at=datetime.now(timezone.utc).isoformat(),
            )
    except Exception as exc:
        print(f"[vault] mine error {file_id}: {exc}")
        set_status(file_id, "error")
