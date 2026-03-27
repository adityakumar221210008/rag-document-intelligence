import faiss
import numpy as np
import pickle
import os
from app.ingestion.embedder import DIM

INDEX_PATH = "faiss.index"
META_PATH = "faiss_meta.pkl"


class VectorStore:
    def __init__(self):
        self.dim = DIM
        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[VectorStore] Loaded {self.index.ntotal} vectors from disk.")
        else:
            # IndexFlatIP = exact inner product search (cosine sim when normalized)
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata: list[dict] = []
            print("[VectorStore] Created fresh index.")

    def add(self, vectors: np.ndarray, chunks: list[dict]):
        """Add embeddings + metadata. Vectors must already be normalized."""
        vectors = vectors.astype("float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.metadata.extend(chunks)
        self._save()

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict]:
        """Return top-k chunks with cosine similarity scores."""
        query_vec = query_vec.astype("float32")
        faiss.normalize_L2(query_vec)
        scores, ids = self.index.search(query_vec, k)
        results = []
        for j, i in enumerate(ids[0]):
            if i != -1:
                results.append({
                    "chunk": self.metadata[i],
                    "score": float(scores[0][j]),
                })
        return results

    def _save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.metadata, f)

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
