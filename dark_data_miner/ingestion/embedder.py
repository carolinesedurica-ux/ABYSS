"""
Embeds DocumentChunks with sentence-transformers or Gemini API and persists them in ChromaDB.
Uses upsert so re-ingesting the same file is idempotent (same chunk_id = same content).
"""
import os
import chromadb
from .models import DocumentChunk

COLLECTION_NAME = "dark_data"
_embedding_model = None
_chroma_client = None

IS_VERCEL = "VERCEL" in os.environ
USE_GEMINI_EMBEDDING = os.getenv("USE_GEMINI_EMBEDDING", "true" if IS_VERCEL else "false").lower() == "true"


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        default_path = "/tmp/chroma_db" if IS_VERCEL else "./data/chroma_db"
        chroma_path = os.getenv("CHROMA_PATH", default_path)
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if USE_GEMINI_EMBEDDING:
        from google import genai
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
        )
        return [emb.values for emb in res.embeddings]
    else:
        model = _get_embedding_model()
        return model.encode(texts, show_progress_bar=True, batch_size=32).tolist()


def embed_and_store(chunks: list[DocumentChunk]) -> None:
    if not chunks:
        return

    collection = _get_collection()
    texts = [c.content for c in chunks]
    embeddings = _embed_texts(texts)

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.to_chroma_metadata() for c in chunks],
    )


def search(query: str, n_results: int = 8) -> list[tuple[DocumentChunk, float]]:
    """Return (chunk, cosine_similarity_score) pairs, highest score first."""
    collection = _get_collection()
    query_embeddings = _embed_texts([query])

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunk = DocumentChunk.from_chroma_hit(doc, meta, score=1 - dist)
            score = 1.0 - dist  # cosine distance → similarity
            hits.append((chunk, score))

    return hits

