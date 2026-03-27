import os
import numpy as np

# --- Choose your backend ---
# Set EMBEDDING_BACKEND=openai in your .env to use OpenAI
# Default: local sentence-transformers (free, no API key needed)

BACKEND = os.getenv("EMBEDDING_BACKEND", "local")


if BACKEND == "openai":
    from openai import OpenAI
    _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = "text-embedding-3-small"
    DIM = 1536

    def _embed(texts: list[str]) -> np.ndarray:
        response = _client.embeddings.create(input=texts, model=MODEL)
        return np.array([e.embedding for e in response.data], dtype="float32")

else:
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    DIM = 384

    def _embed(texts: list[str]) -> np.ndarray:
        return _model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_texts(chunks: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of text chunks in batches."""
    all_vecs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        all_vecs.append(_embed(batch))
    return np.vstack(all_vecs)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, DIM)."""
    return _embed([query])
