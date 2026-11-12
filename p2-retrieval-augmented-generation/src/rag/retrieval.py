"""
Retriever: combines an Embedder and a VectorIndex to find relevant chunks.

This is a bi-encoder retriever: the query and each document chunk are
embedded independently, then compared by cosine similarity. It's fast
enough to run against large indexes because embedding each chunk is
a one-time cost at index time, and the similarity comparison at query
time is just vector math.

The limitation is accuracy. A bi-encoder compares independent
representations and misses nuances that require reading query and
document together. That's what a cross-encoder re-ranker addresses.
See Part 3 for adding a re-ranking stage on top of this retriever.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .chunking import Chunk, Chunker
from .embedding import Embedder
from .index import VectorIndex, SearchResult

logger = logging.getLogger(__name__)


class Retriever:
    """
    Index documents and retrieve relevant chunks for a given query.

    Separating this from the pipeline makes it easy to build retrieval
    evaluation tooling independently of the generation layer.
    """

    def __init__(self, embedder: Embedder, index: VectorIndex) -> None:
        self._embedder = embedder
        self._index = index

    def index_documents(
        self,
        documents: List[str],
        chunker: Chunker,
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Chunk and index a list of documents.

        documents: raw text strings, one per document
        chunker:   the chunking strategy to apply
        sources:   optional list of identifiers (filenames, URLs, IDs) for
                   each document, preserved on every Chunk for traceability

        Returns the number of chunks added to the index.
        """
        if sources is not None and len(sources) != len(documents):
            raise ValueError(
                f"sources length ({len(sources)}) must match documents length ({len(documents)})"
            )

        source_list = sources or [""] * len(documents)
        all_chunks: List[Chunk] = []

        for doc, source in zip(documents, source_list):
            chunks = chunker.chunk(doc, source=source)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from %d documents", len(documents))
            return 0

        logger.info(
            "Embedding %d chunks from %d documents (batch call)",
            len(all_chunks),
            len(documents),
        )

        texts = [chunk.text for chunk in all_chunks]
        embeddings = self._embedder.embed_batch(texts)

        for chunk, embedding in zip(all_chunks, embeddings):
            self._index.add(chunk, embedding)

        logger.info("Index now contains %d chunks total", len(self._index))
        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """
        Embed the query and return the top_k most similar chunks with scores.

        The scores are cosine similarity values (0.0-1.0). Check the top score
        against your similarity_threshold before passing results to a generator.
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = self._embedder.embed(query)
        results = self._index.search(query_embedding, top_k=top_k)

        logger.debug(
            "Query '%s...' -> %d results (top score: %.3f)",
            query[:40],
            len(results),
            results[0].score if results else 0.0,
        )
        return results
