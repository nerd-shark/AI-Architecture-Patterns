"""
Tests for the full RAG pipeline.

These tests verify pipeline behavior end-to-end using MockEmbedder and
MockGenerator — no API calls, no network, deterministic results.

Key behaviors under test:
  - Indexing accumulates chunks correctly
  - Sources propagate from documents through to search results
  - The similarity threshold safety check routes to NO_CONTEXT_RESPONSE
  - retrieve() returns results sorted by score, descending
  - Multiple index() calls accumulate (don't reset)
"""

import pytest

from rag.chunking import FixedSizeChunker
from rag.generation import NO_CONTEXT_RESPONSE
from rag.index import InMemoryIndex
from rag.pipeline import RAGPipeline

from .conftest import MockEmbedder, MockGenerator


class TestIndexing:
    def test_index_returns_chunk_count(self, pipeline):
        count = pipeline.index(["A document with enough words to become at least one chunk."])
        assert count >= 1

    def test_chunk_count_increases_after_index(self, pipeline):
        pipeline.index(["First document with content."])
        after_first = pipeline.chunk_count
        pipeline.index(["Second document with more content."])
        assert pipeline.chunk_count > after_first

    def test_multiple_index_calls_accumulate(self, pipeline):
        pipeline.index(["Batch one."])
        pipeline.index(["Batch two."])
        pipeline.index(["Batch three."])
        assert pipeline.chunk_count >= 3

    def test_sources_propagate_to_chunks(self):
        embedder = MockEmbedder()
        index = InMemoryIndex()
        generator = MockGenerator()
        p = RAGPipeline(
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
            embedder=embedder,
            index=index,
            generator=generator,
        )
        p.index(
            ["Policy document content here with enough words."],
            sources=["returns-policy.txt"],
        )
        results = p.retrieve("policy content")
        assert all(r.chunk.source == "returns-policy.txt" for r in results)

    def test_sources_length_mismatch_raises(self, pipeline):
        with pytest.raises(ValueError, match="sources length"):
            pipeline.index(["doc one", "doc two"], sources=["only-one-source.txt"])


class TestRetrieval:
    def test_retrieve_returns_results(self, pipeline):
        pipeline.index(["Some document content for retrieval testing purposes."])
        results = pipeline.retrieve("document content")
        assert len(results) > 0

    def test_results_sorted_by_score_descending(self, pipeline):
        pipeline.index(["Content about refunds and returns.", "Content about shipping times."])
        results = pipeline.retrieve("refund policy")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_valid_range(self, pipeline):
        pipeline.index(["Test content for scoring validation."])
        results = pipeline.retrieve("test content")
        assert all(-1.0 <= r.score <= 1.0 for r in results)

    def test_empty_index_returns_empty_results(self, pipeline):
        # Pipeline fixture has no documents indexed
        results = pipeline.retrieve("any query")
        assert results == []

    def test_empty_query_raises(self, pipeline):
        pipeline.index(["Some document."])
        with pytest.raises(ValueError, match="empty"):
            pipeline.retrieve("")


class TestSimilarityThreshold:
    def test_below_threshold_returns_no_context_response(self):
        """
        Core safety mechanism: when no chunk exceeds the threshold,
        the pipeline should return NO_CONTEXT_RESPONSE rather than
        generating from poor context.
        """
        embedder = MockEmbedder()
        index = InMemoryIndex()
        # High threshold: will reject everything MockEmbedder returns
        generator = MockGenerator(similarity_threshold=0.9999)
        p = RAGPipeline(
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
            embedder=embedder,
            index=index,
            generator=generator,
        )
        p.index(["Some document content."])
        answer = p.query("completely unrelated query about nothing in the documents")
        assert answer == NO_CONTEXT_RESPONSE

    def test_above_threshold_generates_answer(self):
        """When nothing blocks retrieval, the generator should produce an answer."""
        embedder = MockEmbedder()
        index = InMemoryIndex()
        # threshold=-1.1 accepts any cosine similarity value (range is [-1, 1]).
        # Mock embeddings use sin(), so scores can be negative; this tests that
        # the pipeline produces an answer when the threshold does not block it.
        generator = MockGenerator(similarity_threshold=-1.1)
        p = RAGPipeline(
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
            embedder=embedder,
            index=index,
            generator=generator,
        )
        p.index(["Document with relevant content about the topic."])
        answer = p.query("relevant content")
        assert answer != NO_CONTEXT_RESPONSE
        assert len(answer) > 0

    def test_empty_index_always_returns_no_context(self):
        embedder = MockEmbedder()
        index = InMemoryIndex()
        generator = MockGenerator(similarity_threshold=0.0)
        p = RAGPipeline(
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
            embedder=embedder,
            index=index,
            generator=generator,
        )
        # No documents indexed
        answer = p.query("any question at all")
        assert answer == NO_CONTEXT_RESPONSE


class TestQuery:
    def test_query_returns_string(self, pipeline):
        pipeline.index(["Document content for query testing."])
        result = pipeline.query("document content")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_query_passes_results_to_generator(self):
        embedder = MockEmbedder()
        index = InMemoryIndex()
        generator = MockGenerator(similarity_threshold=0.0)
        p = RAGPipeline(
            chunker=FixedSizeChunker(chunk_size=50, overlap=0),
            embedder=embedder,
            index=index,
            generator=generator,
        )
        p.index(["Some test document content here."])
        p.query("test query")
        assert generator.last_query == "test query"
        assert generator.last_results is not None
        assert len(generator.last_results) > 0
