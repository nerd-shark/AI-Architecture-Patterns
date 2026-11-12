"""
Basic RAG example: index a small knowledge base and answer questions.

Demonstrates the full pipeline with real OpenAI API calls.
Set OPENAI_API_KEY in your environment before running.

Usage:
    pip install -e ".[dev]"
    export OPENAI_API_KEY=sk-...
    python examples/basic_rag.py
"""

import logging

from rag import RAGPipeline
from rag.config import ChunkingConfig, RAGConfig, RetrievalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s: %(message)s",
)

KNOWLEDGE_BASE = [
    """
    Refund Policy — Effective January 2024

    We offer full refunds within 30 days of purchase for physical products.
    Items must be returned in original, undamaged packaging with all accessories included.
    Shipping costs for returns are the customer's responsibility unless the item arrived damaged.

    Digital products (software licenses, downloadable content, subscription services)
    are non-refundable once activated or downloaded, unless the product is defective.
    For defective digital products, contact support within 7 days of purchase.
    """,
    """
    Shipping Policy — Effective January 2024

    Standard shipping takes 5-7 business days within the continental United States.
    Expedited shipping (2-3 business days) is available for an additional fee.
    International shipping is available to 45 countries; delivery times vary by destination.

    Orders over $50 qualify for free standard shipping.
    Orders are processed within 1 business day of placement.
    You will receive a tracking number by email once your order ships.
    """,
    """
    Privacy Policy — Summary

    We collect only the information necessary to process your orders and improve our service.
    We do not sell your personal data to third parties.
    You may request deletion of your account data at any time by contacting privacy@example.com.
    Data is stored on encrypted servers and retained for no longer than 7 years.
    """,
]

SOURCES = [
    "refund-policy.txt",
    "shipping-policy.txt",
    "privacy-policy.txt",
]

QUESTIONS = [
    # In-scope: should retrieve relevant context and answer
    "Can I return a digital product I already downloaded?",
    "How long does standard shipping take?",
    "Do you sell my personal data to third parties?",
    # Out-of-scope: should trigger the similarity threshold and return no-context response
    "What is the weather forecast for tomorrow?",
]


def main() -> None:
    config = RAGConfig(
        chunking=ChunkingConfig(strategy="sentence", chunk_size=100),
        retrieval=RetrievalConfig(top_k=3, similarity_threshold=0.70),
    )

    pipeline = RAGPipeline.from_config(config)

    print("Building knowledge base index...")
    chunk_count = pipeline.index(KNOWLEDGE_BASE, sources=SOURCES)
    print(f"Indexed {chunk_count} chunks from {len(KNOWLEDGE_BASE)} documents.\n")
    print("=" * 60)

    for question in QUESTIONS:
        print(f"\nQ: {question}")
        answer = pipeline.query(question)
        print(f"A: {answer}")

        # Show retrieval scores so you can see the threshold in action
        results = pipeline.retrieve(question)
        if results:
            top = results[0]
            print(
                f"   [top match: score={top.score:.3f}, source='{top.chunk.source}']"
            )
        print("-" * 60)


if __name__ == "__main__":
    main()
