"""
Shared test fixtures and mock implementations.

MockEmbedder and MockGenerator let the full pipeline run in tests
without making any API calls. The embeddings are deterministic and
intentionally simple — the goal is to test pipeline logic (threshold
behavior, source propagation, chunk accumulation), not embedding quality.
"""

import hashlib
import math
from typing import List, Optional

import pytest

from rag.chunking import FixedSizeChunker, SentenceChunker
from rag.embedding import Embedder, Embedding
from rag.generation import NO_CONTEXT_RESPONSE
from rag.index import InMemoryIndex, SearchResult
from rag.pipeline import RAGPipeline


class MockEmbedder(Embedder):
    """
    Deterministic embedder for tests: no API calls, no network.

    Produces 4-dimensional embeddings based on the first 20 characters
    of the text. Texts with the same prefix get the same embedding,
    which is good enough to test similarity scoring behavior.
    """

    DIM = 4

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        seed = int(hashlib.md5(text.lower()[:20].encode()).hexdigest(), 16)
        return [
            math.sin(seed * 0.001 * (i + 1))
            for i in range(self.DIM)
        ]

    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        return [self.embed(t) for t in texts]


class MockGenerator:
    """
    Generator stub for tests: records calls, respects the threshold.
    """

    def __init__(self, similarity_threshold: float = 0.0) -> None:
        self.similarity_threshold = similarity_threshold
        self.last_query: Optional[str] = None
        self.last_results: Optional[List[SearchResult]] = None
        self.call_count = 0

    def generate(self, query: str, results: List[SearchResult]) -> str:
        self.last_query = query
        self.last_results = results
        self.call_count += 1

        if not results or results[0].score < self.similarity_threshold:
            return NO_CONTEXT_RESPONSE
        return f"Mock answer for: {query!r} (from {len(results)} chunks)"


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def in_memory_index() -> InMemoryIndex:
    return InMemoryIndex()


@pytest.fixture
def mock_generator() -> MockGenerator:
    return MockGenerator(similarity_threshold=0.5)


@pytest.fixture
def pipeline(mock_embedder, in_memory_index, mock_generator) -> RAGPipeline:
    """A fully wired pipeline with no external dependencies."""
    return RAGPipeline(
        chunker=SentenceChunker(target_size=50, overlap_sentences=0),
        embedder=mock_embedder,
        index=in_memory_index,
        generator=mock_generator,
        top_k=3,
    )
