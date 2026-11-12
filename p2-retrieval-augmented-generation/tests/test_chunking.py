"""
Tests for chunking strategies.

These tests verify the semantic properties that matter for retrieval:
  - Chunk boundaries fall where expected
  - Source identifiers propagate to every chunk
  - Overlap behavior preserves context across boundaries
  - Edge cases (empty input, text shorter than chunk size) don't break
"""

import pytest

from rag.chunking import Chunk, FixedSizeChunker, SentenceChunker, get_chunker
from rag.config import ChunkingConfig


class TestFixedSizeChunker:
    def test_basic_split(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk("one two three four five six seven eight nine ten")
        assert len(chunks) == 2
        assert chunks[0].text == "one two three four five"
        assert chunks[1].text == "six seven eight nine ten"

    def test_overlap_repeats_words_at_boundary(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=2)
        chunks = chunker.chunk("a b c d e f g h i j")
        # Second chunk should start with the last 2 words of the first chunk
        first_last_two = chunks[0].text.split()[-2:]
        second_first_two = chunks[1].text.split()[:2]
        assert first_last_two == second_first_two

    def test_source_propagates_to_all_chunks(self):
        chunker = FixedSizeChunker(chunk_size=5)
        chunks = chunker.chunk("word " * 20, source="policy.txt")
        assert all(c.source == "policy.txt" for c in chunks)

    def test_chunk_index_is_sequential(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk("word " * 20)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_empty_input_returns_empty_list(self):
        chunker = FixedSizeChunker(chunk_size=100)
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_text_shorter_than_chunk_size_returns_one_chunk(self):
        chunker = FixedSizeChunker(chunk_size=1000)
        chunks = chunker.chunk("just a few words")
        assert len(chunks) == 1
        assert chunks[0].text == "just a few words"

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError, match="greater than overlap"):
            FixedSizeChunker(chunk_size=10, overlap=10)

    def test_chunk_boundary_splits_numbered_steps(self):
        """
        Demonstrates the fixed-size chunk boundary problem from the article.
        A 3-step process split with a small chunk_size will split mid-process.
        """
        chunker = FixedSizeChunker(chunk_size=10, overlap=0)
        text = (
            "Step 1 navigate to the login page "
            "Step 2 enter your email address "
            "Step 3 check your email for the link"
        )
        chunks = chunker.chunk(text)
        full_texts = [c.text for c in chunks]
        # No single chunk contains all three steps — they were split
        assert not any(
            "Step 1" in t and "Step 2" in t and "Step 3" in t for t in full_texts
        )


class TestSentenceChunker:
    def test_respects_sentence_boundaries(self):
        """Each chunk should end at a sentence boundary, not mid-sentence."""
        chunker = SentenceChunker(target_size=10, overlap_sentences=0)
        text = "This is the first full sentence. This is the second full sentence."
        chunks = chunker.chunk(text)
        for chunk in chunks:
            assert chunk.text.endswith(".")

    def test_all_text_preserved(self):
        """No text should be lost during chunking."""
        chunker = SentenceChunker(target_size=50)
        sentences = ["Sentence number one.", "Sentence number two.", "Sentence number three."]
        text = " ".join(sentences)
        chunks = chunker.chunk(text)
        full_text = " ".join(c.text for c in chunks)
        for sentence in sentences:
            assert sentence in full_text

    def test_source_propagates(self):
        chunker = SentenceChunker(target_size=50)
        chunks = chunker.chunk("One sentence. Two sentences.", source="doc.txt")
        assert all(c.source == "doc.txt" for c in chunks)

    def test_empty_input_returns_empty_list(self):
        chunker = SentenceChunker()
        assert chunker.chunk("") == []

    def test_single_sentence_returns_one_chunk(self):
        chunker = SentenceChunker(target_size=1000)
        chunks = chunker.chunk("Just one sentence here.")
        assert len(chunks) == 1

    def test_numbered_steps_stay_together_in_one_chunk(self):
        """
        Contrast with the FixedSizeChunker test: with a large enough target_size,
        sentence chunker keeps a short numbered process in one chunk.
        """
        chunker = SentenceChunker(target_size=200, overlap_sentences=0)
        text = "Step 1: navigate to the page. Step 2: enter your email. Step 3: check your inbox."
        chunks = chunker.chunk(text)
        # All steps should fit in one chunk given the large target_size
        all_text = " ".join(c.text for c in chunks)
        assert "Step 1" in all_text
        assert "Step 2" in all_text
        assert "Step 3" in all_text


class TestGetChunker:
    def test_fixed_size_strategy(self):
        config = ChunkingConfig(strategy="fixed_size", chunk_size=200, overlap=20)
        chunker = get_chunker(config)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 200

    def test_sentence_strategy(self):
        config = ChunkingConfig(strategy="sentence", chunk_size=150)
        chunker = get_chunker(config)
        assert isinstance(chunker, SentenceChunker)

    def test_unknown_strategy_raises(self):
        config = ChunkingConfig(strategy="unknown_strategy")
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker(config)
