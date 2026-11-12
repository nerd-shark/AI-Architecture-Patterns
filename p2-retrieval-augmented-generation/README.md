# Part 2: Retrieval-Augmented Generation (RAG)

Production implementation for the [RAG article](link) in the AI Architecture Patterns series.

**GitHub**: [github.com/nerd-shark/AI-Architecture-Patterns](https://github.com/nerd-shark/AI-Architecture-Patterns) — `p2-retrieval-augmented-generation/`

## What This Implements

The full two-phase RAG pipeline with clean, swappable components:

| Component | Interface | Default Implementation | Swap For |
|-----------|-----------|----------------------|----------|
| Chunking | `Chunker` | `SentenceChunker` | Structural chunker, spacy-based chunker |
| Embedding | `Embedder` | `OpenAIEmbedder` | sentence-transformers, Cohere, local models |
| Vector Index | `VectorIndex` | `InMemoryIndex` | Pinecone, Qdrant, pgvector, ChromaDB |
| Generation | `Generator` | OpenAI chat completion | Anthropic, local LLM |

Each interface is a single abstract class. Swap any piece by subclassing the interface and passing your implementation to `RAGPipeline(...)`.

## Setup

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...
```

## Run the Tests (no API key needed)

```bash
pytest
```

Tests use `MockEmbedder` and `MockGenerator` and make no external calls.

## Run the Example

```bash
python examples/basic_rag.py
```

Indexes a three-document knowledge base (refund policy, shipping policy, privacy policy) and runs four questions through the pipeline. The last question is out of scope and demonstrates the similarity threshold safety check returning a "no information" response.

## Configuration

Everything flows from `RAGConfig`:

```python
from rag import RAGPipeline
from rag.config import RAGConfig, ChunkingConfig, RetrievalConfig

config = RAGConfig(
    chunking=ChunkingConfig(
        strategy="sentence",   # or "fixed_size"
        chunk_size=200,        # target words per chunk
    ),
    retrieval=RetrievalConfig(
        top_k=5,
        similarity_threshold=0.75,  # tune this for your domain
    ),
)
pipeline = RAGPipeline.from_config(config)
pipeline.index(["Your documents..."])
answer = pipeline.query("Your question")
```

## Swapping the Vector Store

`InMemoryIndex` is for development and small knowledge bases (~10,000 chunks).
For production, subclass `VectorIndex` and pass your implementation directly:

```python
from myapp.storage import PineconeIndex  # your VectorIndex subclass
from rag.chunking import get_chunker
from rag.embedding import get_embedder
from rag.generation import Generator
from rag.config import RAGConfig

config = RAGConfig()
pipeline = RAGPipeline(
    chunker=get_chunker(config.chunking),
    embedder=get_embedder(config.embedding),
    index=PineconeIndex(api_key="...", index_name="my-index"),
    generator=Generator(similarity_threshold=0.75),
)
```

## Swapping the Embedding Model

`OpenAIEmbedder` calls the OpenAI API. For a local model, subclass `Embedder`:

```python
from rag.embedding import Embedder, Embedding
from sentence_transformers import SentenceTransformer

class LocalEmbedder(Embedder):
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> Embedding:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[Embedding]:
        return self._model.encode(texts).tolist()
```

## Similarity Threshold Tuning

The default `similarity_threshold=0.75` is a starting point, not a universal value.
To find the right threshold for your knowledge base:

1. Collect 50-100 real queries you expect the system to answer
2. Run `pipeline.retrieve(query)` for each and inspect the top score
3. Find the score that separates "clearly relevant" from "clearly not relevant"
4. Set the threshold just below that boundary

Scores vary by embedding model, chunk size, and domain vocabulary. Tune
against your data before deploying.

## Project Structure

```
p2-retrieval-augmented-generation/
├── src/rag/
│   ├── config.py       — All tuneable parameters
│   ├── chunking.py     — FixedSizeChunker, SentenceChunker, Chunker ABC
│   ├── embedding.py    — OpenAIEmbedder, Embedder ABC
│   ├── index.py        — InMemoryIndex, VectorIndex ABC, SearchResult
│   ├── retrieval.py    — Retriever (bi-encoder)
│   ├── generation.py   — Generator with similarity threshold check
│   └── pipeline.py     — RAGPipeline (top-level coordinator)
├── tests/
│   ├── conftest.py     — MockEmbedder, MockGenerator, shared fixtures
│   ├── test_chunking.py
│   └── test_pipeline.py
└── examples/
    └── basic_rag.py
```

## What's Not Here (Intentionally)

This implementation covers the core RAG pattern described in Part 2.
The following are covered in later parts of the series:

- **Re-ranking with a cross-encoder** — Part 3: Beyond Naive RAG
- **Hybrid lexical + semantic search** — Part 3
- **Graph retrieval for multi-hop queries** — Part 3
- **Semantic caching** — Part 17
- **Context window management** — Part 21
