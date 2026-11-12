"""
RAGPipeline: the top-level coordinator for the full RAG pattern.

This class wires the four components together and exposes the two
operations that callers actually need: index() and query().

The component design is intentional: each piece has one job and a
clean interface, so you can swap implementations without touching
the pipeline logic. Replace InMemoryIndex with Pinecone. Replace
OpenAIEmbedder with a local sentence-transformers model. Replace
the Generator with a different provider. None of those changes
require modifying RAGPipeline itself.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .config import RAGConfig
from .chunking import Chunker, get_chunker
from .embedding import Embedder, get_embedder
from .generation import Generator
from .index import InMemoryIndex, SearchResult, VectorIndex
from .retrieval import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    The complete retrieval-augmented generation pipeline.

    index() and query() are the two public operations. Everything else
    is an implementation detail.

    Quick start:
        pipeline = RAGPipeline.from_config()
        pipeline.index(["Your documents go here..."])
        answer = pipeline.query("What do the documents say about X?")

    Custom configuration:
        from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig
        config = RAGConfig(
            chunking=ChunkingConfig(strategy="sentence", chunk_size=200),
            retrieval=RetrievalConfig(similarity_threshold=0.80),
        )
        pipeline = RAGPipeline.from_config(config)

    Swapping the vector store:
        from myapp.storage import PineconeIndex  # your VectorIndex subclass
        pipeline = RAGPipeline(
            chunker=get_chunker(config.chunking),
            embedder=get_embedder(config.embedding),
            index=PineconeIndex(...),
            generator=Generator(...),
            top_k=config.retrieval.top_k,
        )
    """

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        index: VectorIndex,
        generator: Generator,
        top_k: int = 3,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._index = index
        self._generator = generator
        self._top_k = top_k
        self._retriever = Retriever(embedder=embedder, index=index)

    @classmethod
    def from_config(cls, config: Optional[RAGConfig] = None) -> "RAGPipeline":
        """
        Build a pipeline from a RAGConfig.

        Uses InMemoryIndex. For production, construct the pipeline directly
        and pass your persistent VectorIndex implementation.
        """
        config = config or RAGConfig()

        chunker = get_chunker(config.chunking)
        embedder = get_embedder(config.embedding)
        index = InMemoryIndex()
        generator = Generator(
            model=config.generation.model,
            api_key=config.generation.api_key,
            similarity_threshold=config.retrieval.similarity_threshold,
            max_tokens=config.generation.max_tokens,
            temperature=config.generation.temperature,
        )

        return cls(
            chunker=chunker,
            embedder=embedder,
            index=index,
            generator=generator,
            top_k=config.retrieval.top_k,
        )

    def index(
        self,
        documents: List[str],
        sources: Optional[List[str]] = None,
    ) -> int:
        """
        Chunk and index a list of documents.

        documents: list of raw text strings
        sources:   optional list of document identifiers (filenames, URLs)
                   preserved on every chunk and surfaced in search results

        Returns the number of chunks added to the index.
        """
        return self._retriever.index_documents(
            documents, self._chunker, sources=sources
        )

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        Return raw retrieval results for a query, with similarity scores.

        Useful for evaluation: inspect scores and retrieved text before
        the generation step to understand retrieval quality independently.
        """
        return self._retriever.retrieve(query, top_k=self._top_k)

    def query(self, query: str) -> str:
        """
        Full pipeline: retrieve relevant chunks, then generate a grounded answer.

        If no retrieved chunk exceeds the similarity threshold, returns a
        "no information" response rather than generating from poor context.
        """
        results = self.retrieve(query)
        return self._generator.generate(query, results)

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the index."""
        return len(self._index)
