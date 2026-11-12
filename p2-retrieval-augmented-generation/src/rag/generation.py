"""
Generator: produce a grounded answer from retrieved chunks.

The similarity threshold check is the core safety mechanism here.
Without it, when nothing relevant is retrieved, the model receives
marginally-related chunks and fills the gap from its training data —
producing a confident, fluent, wrong answer with no indication that
retrieval failed. With the check, it acknowledges the gap instead.

Tune SIMILARITY_THRESHOLD in RetrievalConfig against real queries for
your domain. A threshold that works well on one knowledge base may need
adjustment on another.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .index import SearchResult

logger = logging.getLogger(__name__)


# Default system prompt. Override by passing system_prompt to Generator.__init__
# if your use case needs different grounding instructions.
DEFAULT_SYSTEM_PROMPT = """You are a precise assistant that answers questions \
using only the provided context documents.

Rules:
- Base your answer entirely on the context provided. Do not draw on outside knowledge.
- If the context doesn't contain enough information to answer the question, say so clearly.
- Reference the relevant part of the context when it helps the answer.
- Keep answers accurate and concise."""

NO_CONTEXT_RESPONSE = (
    "The retrieved context doesn't contain enough information to answer this question "
    "reliably. Please consult the source documents directly, or try rephrasing."
)


class Generator:
    """
    Generate a grounded answer from a query and retrieved search results.

    The similarity_threshold controls the "no information" branch:
    if the top retrieved result is below the threshold, the generator
    returns NO_CONTEXT_RESPONSE rather than generating from poor context.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        similarity_threshold: float = 0.75,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def generate(self, query: str, results: List[SearchResult]) -> str:
        """
        Generate a grounded answer from the query and retrieved chunks.

        Returns NO_CONTEXT_RESPONSE if the top result is below the similarity
        threshold. This is the "no information" branch that prevents the model
        from answering out-of-scope questions from parametric knowledge.
        """
        if not results:
            logger.info("No results retrieved for query; returning no-context response")
            return NO_CONTEXT_RESPONSE

        top_score = results[0].score
        if top_score < self.similarity_threshold:
            logger.info(
                "Top score %.3f below threshold %.3f; returning no-context response",
                top_score,
                self.similarity_threshold,
            )
            return NO_CONTEXT_RESPONSE

        context = self._format_context(results)

        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content

    def _format_context(self, results: List[SearchResult]) -> str:
        """
        Format retrieved chunks as a numbered context block.

        Source names are included when available so the model can reference
        specific documents and the user can verify the answer.
        """
        parts = []
        for i, result in enumerate(results, 1):
            source_label = f" [{result.chunk.source}]" if result.chunk.source else ""
            parts.append(f"[Context {i}{source_label}]\n{result.chunk.text}")
        return "\n\n".join(parts)
