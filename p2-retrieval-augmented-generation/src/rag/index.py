"""
Vector index abstraction and in-memory implementation.

The VectorIndex interface is the second major swap point. InMemoryIndex
is suitable for development and small knowledge bases (up to ~10,000 chunks).
For production, replace it with a persistent vector store without changing
any other component.

Production alternatives:
  - Pinecone  (managed, serverless, good for variable load)
  - Qdrant    (open source, self-hosted or cloud, strong filtering)
  - Weaviate  (open source, graph-aware, strong schema support)
  - pgvector  (Postgres extension, good if you're already on Postgres)
  - ChromaDB  (embedded, good for local/edge deployments)

All of them expose a compatible add/search interface. The swap is
straightforward: implement VectorIndex with your chosen store's client.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from .chunking import Chunk
from .embedding import Embedding

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A retrieved chunk with its similarity score."""

    chunk: Chunk
    score: float
    # Cosine similarity: range is -1.0 to 1.0.
    # In practice, real embedding models (OpenAI, sentence-transformers) return
    # scores in roughly [0.0, 1.0] for meaningful text. The similarity_threshold
    # in RetrievalConfig should be set relative to what your chosen embedder produces.

    def __repr__(self) -> str:
        preview = self.chunk.text[:70] + "..." if len(self.chunk.text) > 70 else self.chunk.text
        return f"SearchResult(score={self.score:.3f}, source={self.chunk.source!r}, text={preview!r})"


class VectorIndex(ABC):
    """Interface for adding chunks and searching by embedding similarity."""

    @abstractmethod
    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        """Add a chunk and its embedding to the index."""
        ...

    @abstractmethod
    def search(self, query_embedding: Embedding, top_k: int = 3) -> List[SearchResult]:
        """Return the top_k most similar chunks, sorted by score descending."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Number of chunks currently in the index."""
        ...


class InMemoryIndex(VectorIndex):
    """
    In-memory vector index using numpy cosine similarity.

    All embeddings are stored in RAM and searched with a full matrix multiply.
    This is O(n) per query, which is acceptable up to ~10,000 chunks (search
    time stays under 10ms on a standard laptop at that scale).

    Beyond that, consider approximate nearest-neighbor indexes (HNSW, IVF)
    available in the production stores listed in the module docstring.
    """

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._embeddings: List[Embedding] = []

    def add(self, chunk: Chunk, embedding: Embedding) -> None:
        self._chunks.append(chunk)
        self._embeddings.append(embedding)

    def search(self, query_embedding: Embedding, top_k: int = 3) -> List[SearchResult]:
        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        matrix = np.array(self._embeddings, dtype=np.float32)

        # Cosine similarity: normalize both vectors, then dot product.
        # Adding a small epsilon avoids division by zero on zero vectors.
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        normalized = matrix / row_norms
        scores = normalized @ q_norm  # shape: (n_chunks,)

        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            SearchResult(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all indexed chunks. Useful between test runs."""
        self._chunks = []
        self._embeddings = []
