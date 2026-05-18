"""
Embeds DocumentChunks with sentence-transformers and persists them in ChromaDB.
Uses upsert so re-ingesting the same file is idempotent (same chunk_id = same content).
"""
import os
from sentence_transformers import SentenceTransformer
import chromadb
from .models import DocumentChunk

COLLECTION_NAME = "dark_data"
_embedding_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _get_collection() -> chromadb.Collection:
    global _chroma_client
    if _chroma_client is None:
        chroma_path = os.getenv("CHROMA_PATH", "./data/chroma_db")
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_and_store(chunks: list[DocumentChunk]) -> None:
    if not chunks:
        return

    model = _get_embedding_model()
    collection = _get_collection()

    texts = [c.content for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[c.to_chroma_metadata() for c in chunks],
    )


def search(query: str, n_results: int = 8) -> list[tuple[DocumentChunk, float]]:
    """Return (chunk, cosine_similarity_score) pairs, highest score first."""
    model = _get_embedding_model()
    collection = _get_collection()

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunk = DocumentChunk.from_chroma_hit(doc, meta, score=1 - dist)
        score = 1.0 - dist  # cosine distance → similarity
        hits.append((chunk, score))

    return hits
