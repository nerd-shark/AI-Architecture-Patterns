"""
RAG: Retrieval-Augmented Generation — AI Architecture Patterns, Part 2.

Public API:
    from rag import RAGPipeline, RAGConfig

For custom configuration:
    from rag import RAGConfig, ChunkingConfig, RetrievalConfig
    config = RAGConfig(
        chunking=ChunkingConfig(strategy="sentence"),
        retrieval=RetrievalConfig(similarity_threshold=0.80),
    )
    pipeline = RAGPipeline.from_config(config)
    pipeline.index(documents)
    answer = pipeline.query("Your question")
"""

from .chunking import Chunk, Chunker, FixedSizeChunker, SentenceChunker
from .config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    RAGConfig,
    RetrievalConfig,
)
from .embedding import Embedder, Embedding, OpenAIEmbedder
from .generation import NO_CONTEXT_RESPONSE, Generator
from .index import InMemoryIndex, SearchResult, VectorIndex
from .pipeline import RAGPipeline
from .retrieval import Retriever

__all__ = [
    # Main entry points
    "RAGPipeline",
    "RAGConfig",
    # Configuration
    "ChunkingConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "GenerationConfig",
    # Abstractions (for custom implementations)
    "Chunker",
    "Embedder",
    "VectorIndex",
    # Data types
    "Chunk",
    "Embedding",
    "SearchResult",
    # Concrete implementations
    "FixedSizeChunker",
    "SentenceChunker",
    "OpenAIEmbedder",
    "InMemoryIndex",
    "Retriever",
    "Generator",
    # Constants
    "NO_CONTEXT_RESPONSE",
]
