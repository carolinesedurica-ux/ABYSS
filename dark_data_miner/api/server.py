import shutil
import tempfile
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent.parent / "omno")

from ..agents.ingestion_agent import IngestionState, build_ingestion_graph
from ..agents.query_agent import QueryState, build_query_graph
from ..agents.router_agent import PipelineState, build_pipeline
from .vault_router import router as vault_router

STATIC = Path(__file__).parent.parent.parent / "static"

app = FastAPI(title="A.B.Y.S.S. — Automated Business Intelligence & Shadow-data Scanner", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.include_router(vault_router)


@app.get("/")
def ui():
    return FileResponse(str(STATIC / "index.html"))

SUPPORTED_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".pdf"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── /run — full agentic pipeline ──────────────────────────────────────────────

class RunRequest(BaseModel):
    input: str                   # natural language — router decides intent

@app.post("/run")
def run(req: RunRequest):
    pipeline = build_pipeline()
    result: PipelineState = pipeline.invoke({
        "user_input": req.input,
        "intent": "query",
        "file_paths": [],
        "answer": "",
        "citations": [],
        "processed": [],
        "total_chunks": 0,
        "errors": [],
    })
    return {
        "intent": result["intent"],
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "processed": result.get("processed", []),
        "errors": result.get("errors", []),
    }


# ── /ingest — direct file ingestion ──────────────────────────────────────────

@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    tmp = Path(tempfile.mkdtemp())
    saved: list[str] = []

    for upload in files:
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in SUPPORTED_EXTS:
            raise HTTPException(400, f"Unsupported file type: {suffix}")
        dest = tmp / upload.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        saved.append(str(dest))

    graph = build_ingestion_graph()
    result: IngestionState = graph.invoke({
        "file_paths": saved,
        "processed": [],
        "errors": [],
        "total_chunks": 0,
    })

    return {
        "processed": [Path(p).name for p in result["processed"]],
        "errors": result["errors"],
        "total_chunks": result["total_chunks"],
    }


# ── /query — direct agentic query ────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query(req: QueryRequest):
    try:
        graph = build_query_graph()
        result: QueryState = graph.invoke({
            "original_query": req.query,
            "query": req.query,
            "chunks": [],
            "scores": [],
            "retrieval_sufficient": False,
            "retry_count": 0,
            "answer": "",
            "citations": [],
        })
        return {"answer": result["answer"], "citations": result["citations"]}
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[query error] {exc}\n{tb}")
        raise HTTPException(500, detail=str(exc))


# ── /sources — list indexed files ─────────────────────────────────────────────

@app.get("/sources")
def sources():
    pipeline = build_pipeline()
    result: PipelineState = pipeline.invoke({
        "user_input": "list",
        "intent": "list",
        "file_paths": [],
        "answer": "",
        "citations": [],
        "processed": [],
        "total_chunks": 0,
        "errors": [],
    })
    return {"answer": result["answer"]}
