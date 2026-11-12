"""
Embedding abstraction and OpenAI implementation.

The Embedder interface is the first major swap point in the pipeline.
Swap OpenAIEmbedder for a local model embedder (e.g. sentence-transformers)
without touching any other component.

Domain mismatch is the silent failure here: a general model trained on web
text may not represent your domain's vocabulary well. If retrieval quality
is poor despite good chunking, evaluate a domain-tuned model using MTEB
or your own labeled query-document pairs before concluding the pipeline
is misconfigured.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

# Type alias: an embedding is a list of floats.
# text-embedding-3-small produces 1,536 floats; -large produces 3,072.
Embedding = List[float]


class Embedder(ABC):
    """
    Convert text to an embedding vector.

    Implementations must be stateless with respect to the text being embedded:
    embed("foo") should always return the same vector regardless of call order.
    """

    @abstractmethod
    def embed(self, text: str) -> Embedding:
        """Embed a single string. Raises ValueError for empty input."""
        ...

    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Embed a list of strings, returning embeddings in the same order.

        Default implementation calls embed() sequentially. Subclasses should
        override this to use the provider's native batch API, which is
        significantly more efficient for indexing large document sets.
        """
        return [self.embed(t) for t in texts]


class OpenAIEmbedder(Embedder):
    """
    Embed text using the OpenAI Embeddings API.

    Recommended models:
      text-embedding-3-small  — $0.02/1M tokens, good general-purpose baseline
      text-embedding-3-large  — $0.13/1M tokens, better for domain-specific content

    For domain-specific use cases (legal, medical, code), evaluate against
    your actual query-document pairs before accepting the MTEB leaderboard
    ranking as a proxy for your workload.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model
        logger.info("OpenAIEmbedder initialized with model=%s", model)

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text")
        response = self._client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[Embedding]:
        """
        Batch embedding: one API call for all texts.

        The OpenAI batch API is ~10x more efficient than sequential calls
        for large indexing jobs. Results are returned in input order.
        """
        if not texts:
            return []

        non_empty = [t for t in texts if t.strip()]
        skipped = len(texts) - len(non_empty)
        if skipped:
            logger.warning(
                "Skipping %d empty strings in batch embed call", skipped
            )

        if not non_empty:
            return []

        response = self._client.embeddings.create(
            input=non_empty, model=self.model
        )
        # API returns embeddings in the same order as input
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def get_embedder(config) -> Embedder:
    """Factory: construct the embedder described by an EmbeddingConfig."""
    if config.provider == "openai":
        return OpenAIEmbedder(model=config.model, api_key=config.api_key)
    raise ValueError(
        f"Unknown embedding provider: {config.provider!r}. "
        f"Expected 'openai'. Add your provider here."
    )
