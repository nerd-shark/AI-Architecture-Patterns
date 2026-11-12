"""
Chunking strategies for the RAG indexing pipeline.

A chunk is the unit of retrieval. Its boundaries decide what context
the model can and cannot see at query time, which is why chunking
is a first-class design decision rather than plumbing.

The three strategies here correspond to the tradeoffs in the article:
  - FixedSizeChunker: fast, semantically blind
  - SentenceChunker:  respects sentence boundaries, better for mixed content
  - (Structural chunking is document-format-specific; add it as needed)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A single retrievable unit of text."""

    text: str
    source: str = ""       # filename, URL, or document ID — preserved through retrieval
    index: int = 0         # position within the source document
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60] + "..." if len(self.text) > 60 else self.text
        return f"Chunk(source={self.source!r}, index={self.index}, text={preview!r})"


class Chunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        """Split text into chunks. Returns an empty list for empty input."""
        ...


class FixedSizeChunker(Chunker):
    """
    Split text into fixed-size word chunks with optional overlap.

    Fast and simple. The tradeoff is semantic blindness: a fixed boundary
    can split a numbered list, a sentence, or a cause-and-effect pair
    at an arbitrary point. Use this when documents are homogeneous in
    structure and you want predictable chunk sizes.

    overlap: words repeated at the start of each new chunk. This reduces
    the chance of losing context at a boundary, but doesn't prevent splits
    from occurring mid-thought.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 50) -> None:
        if chunk_size <= overlap:
            raise ValueError(
                f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        words = text.split()
        if not words:
            return []

        step = self.chunk_size - self.overlap
        chunks: List[Chunk] = []

        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + self.chunk_size]
            if chunk_words:
                chunks.append(
                    Chunk(text=" ".join(chunk_words), source=source, index=i)
                )

        return chunks


class SentenceChunker(Chunker):
    """
    Group sentences into chunks up to a target word count.

    Uses sentence boundaries as natural split points, so related content
    stays together. The tradeoff is variable chunk sizes, which makes
    top-K retrieval scores less comparable across chunks.

    overlap_sentences: number of sentences carried over from the previous
    chunk. Provides continuity without duplicating large amounts of text.
    """

    def __init__(
        self, target_size: int = 400, overlap_sentences: int = 1
    ) -> None:
        self.target_size = target_size
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences at '.', '!', '?' boundaries.

        For production use, replace this with spacy's sentencizer or
        nltk.sent_tokenize for better handling of abbreviations, decimals,
        and multi-language content.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, source: str = "") -> List[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current: List[str] = []
        current_word_count = 0
        chunk_index = 0

        for sentence in sentences:
            word_count = len(sentence.split())

            if current_word_count + word_count > self.target_size and current:
                chunks.append(
                    Chunk(
                        text=" ".join(current),
                        source=source,
                        index=chunk_index,
                    )
                )
                chunk_index += 1
                # Carry over the last N sentences as overlap
                current = current[-self.overlap_sentences :]
                current_word_count = sum(len(s.split()) for s in current)

            current.append(sentence)
            current_word_count += word_count

        if current:
            chunks.append(
                Chunk(text=" ".join(current), source=source, index=chunk_index)
            )

        return chunks


def get_chunker(config) -> Chunker:
    """
    Factory: construct the chunker described by a ChunkingConfig.

    Add new strategies here as the catalog grows.
    """
    if config.strategy == "fixed_size":
        return FixedSizeChunker(
            chunk_size=config.chunk_size,
            overlap=config.overlap,
        )
    if config.strategy == "sentence":
        return SentenceChunker(
            target_size=config.chunk_size,
            overlap_sentences=config.overlap,
        )
    raise ValueError(
        f"Unknown chunking strategy: {config.strategy!r}. "
        f"Expected 'fixed_size' or 'sentence'."
    )
