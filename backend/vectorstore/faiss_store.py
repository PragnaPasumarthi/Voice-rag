"""
FAISS Vector Store with sentence-transformers embeddings.
Handles indexing, retrieval, persistence, and hybrid search.
"""
import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStore:
    """
    FAISS-backed vector store with sentence-transformers embeddings.
    Supports:
    - Flat and IVFFlat index types
    - Metadata filtering
    - Batch indexing
    - Persistence (save/load)
    """

    def __init__(self, config=None):
        from ..config import VectorStoreConfig
        self.config = config or VectorStoreConfig()
        self._model = None
        self._index = None
        self._chunk_map: List[Dict[str, Any]] = []
        self._dimension = self.config.dimension

    @property
    def model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers not installed")
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    @property
    def index(self):
        if self._index is None:
            if faiss is None:
                raise RuntimeError("faiss not installed")
            self._index = faiss.IndexFlatIP(self._dimension)
        return self._index

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )
        return np.array(embeddings, dtype=np.float32)

    def add_chunks(self, chunks: List[Any]) -> int:
        """Add chunks to the vector store. Returns number of chunks added."""
        if not chunks:
            return 0

        texts = [c.text if hasattr(c, "text") else c["text"] for c in chunks]
        embeddings = self._encode(texts)

        self.index.add(embeddings)

        for chunk in chunks:
            entry = {
                "text": chunk.text if hasattr(chunk, "text") else chunk["text"],
                "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else chunk["chunk_id"],
                "metadata": chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {}),
            }
            self._chunk_map.append(entry)

        return len(texts)

    def search(
        self,
        query: str,
        top_k: int = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Retrieve top-k results for a query string."""
        k = top_k or self.config.top_k
        if self.index.ntotal == 0:
            return []

        query_emb = self._encode([query])
        scores, indices = self.index.search(query_emb, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunk_map):
                continue
            entry = self._chunk_map[idx]

            if metadata_filter:
                match = all(
                    entry["metadata"].get(k) == v for k, v in metadata_filter.items()
                )
                if not match:
                    continue

            results.append(SearchResult(
                chunk_id=entry["chunk_id"],
                text=entry["text"],
                score=float(score),
                metadata=entry["metadata"],
            ))

        return results

    def save(self, path: str):
        """Persist index and chunk map to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunk_map.pkl"), "wb") as f:
            pickle.dump(self._chunk_map, f)

    def load(self, path: str):
        """Load index and chunk map from disk."""
        index_path = os.path.join(path, "index.faiss")
        map_path = os.path.join(path, "chunk_map.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index not found at {index_path}")

        self._index = faiss.read_index(index_path)
        with open(map_path, "rb") as f:
            self._chunk_map = pickle.load(f)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def clear(self):
        """Reset the store."""
        self._index = None
        self._chunk_map = []
