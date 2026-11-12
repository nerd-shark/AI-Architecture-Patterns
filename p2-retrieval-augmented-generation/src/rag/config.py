"""
Configuration for the RAG pipeline.

All tuneable values live here. Nothing is hardcoded in the pipeline itself.
Pass a RAGConfig to RAGPipeline.from_config() to control every component.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkingConfig:
    """Controls how documents are split before indexing."""

    strategy: str = "sentence"
    # "fixed_size" — fast, semantically blind, good for homogeneous content
    # "sentence"   — respects sentence boundaries, better recall on mixed content

    chunk_size: int = 400
    # For fixed_size: target word count per chunk.
    # For sentence: target word count before starting a new chunk.

    overlap: int = 50
    # For fixed_size: words repeated at the start of each new chunk.
    # For sentence: number of sentences carried over as overlap.


@dataclass
class EmbeddingConfig:
    """Controls which embedding model converts text to vectors."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    # text-embedding-3-small: $0.02/1M tokens, 1536 dims — good default
    # text-embedding-3-large: $0.13/1M tokens, 3072 dims — better for domain-specific content

    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    batch_size: int = 100
    # Number of chunks to embed in a single API call. Batching is significantly
    # more efficient than one call per chunk.


@dataclass
class RetrievalConfig:
    """Controls how many chunks to retrieve and how to filter them."""

    top_k: int = 3
    # Number of chunks to retrieve per query. 3-5 is a typical starting point.
    # Increasing this raises context quality ceiling but also adds noise and tokens.

    similarity_threshold: float = 0.75
    # Minimum similarity score (0-1) for a retrieved chunk to be considered relevant.
    # Below this threshold, the pipeline returns a "no information" response rather
    # than passing low-quality context to the generator.
    #
    # Tune this against real queries for your domain:
    #   - Too high: rejects chunks that are genuinely relevant
    #   - Too low:  lets through chunks with weak overlap, enabling "answer anyway" failures


@dataclass
class GenerationConfig:
    """Controls the language model used for answer generation."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    max_tokens: int = 1000
    temperature: float = 0.0
    # 0.0 for deterministic answers from retrieved context.
    # Increase if you want more varied phrasing, but RAG answers should be grounded,
    # not creative — keep this low.


@dataclass
class RAGConfig:
    """
    Top-level configuration for the RAG pipeline.

    Usage:
        config = RAGConfig()                     # all defaults
        config = RAGConfig(
            chunking=ChunkingConfig(strategy="fixed_size", chunk_size=512),
            retrieval=RetrievalConfig(similarity_threshold=0.80),
        )
        pipeline = RAGPipeline.from_config(config)
    """

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
