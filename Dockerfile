FROM python:3.12-slim

WORKDIR /app

# System deps: ffmpeg for audio/video, libgomp for ML libs, gcc for compiled wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY dark_data_miner/ dark_data_miner/
COPY static/ static/
COPY orchestrator.py .

# Persistent data lives here — mount a volume over this path in production
RUN mkdir -p data/chroma_db data/vault/files

EXPOSE 8000

# Defaults — override at runtime via env vars or platform secrets
ENV CHROMA_PATH=/app/data/chroma_db \
    VAULT_PATH=/app/data/vault \
    WHISPER_MODEL=base \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    USE_GEMINI_VISION=true

CMD ["python", "-m", "uvicorn", "dark_data_miner.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
