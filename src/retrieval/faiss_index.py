from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedder import l2_normalize


@dataclass
class SearchResult:
    index: int
    score: float


class VectorIndex:
    def __init__(self, dim: int, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize
        self._vectors: np.ndarray | None = None
        self._faiss_index: Any | None = None
        self.backend = "numpy"
        try:
            import faiss

            self._faiss = faiss
            self._faiss_index = faiss.IndexFlatIP(dim)
            self.backend = "faiss"
        except Exception:
            self._faiss = None

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors with shape (*, {self.dim}), got {vectors.shape}.")
        if self.normalize:
            vectors = l2_normalize(vectors)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        if self._faiss_index is not None:
            self._faiss_index.add(vectors)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self._vectors is None or len(self._vectors) == 0:
            return []

        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Expected query dimension {self.dim}, got {query.shape[1]}.")
        if self.normalize:
            query = l2_normalize(query)

        top_k = min(max(1, int(top_k)), len(self._vectors))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, top_k)
            return [
                SearchResult(index=int(index), score=float(score))
                for index, score in zip(indices[0], scores[0])
                if int(index) >= 0
            ]

        scores = self._vectors @ query[0]
        best = np.argsort(-scores)[:top_k]
        return [SearchResult(index=int(index), score=float(scores[index])) for index in best]

    def save_vectors(self, path: Path) -> None:
        if self._vectors is None:
            raise ValueError("No vectors have been added to the index.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self._vectors, dim=self.dim, normalize=self.normalize)

    @classmethod
    def load_vectors(cls, path: Path) -> "VectorIndex":
        data = np.load(path, allow_pickle=False)
        dim = int(data["dim"])
        normalize = bool(data["normalize"])
        index = cls(dim=dim, normalize=normalize)
        index.add(np.asarray(data["vectors"], dtype="float32"))
        return index

