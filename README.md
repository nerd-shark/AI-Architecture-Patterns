# AI Architecture Patterns — Implementation Code

Companion implementations for the [AI Architecture Patterns](https://medium.com/gitconnected/ai-systems-have-their-own-pattern-language-6862a5300172) series on Medium.

**GitHub**: [github.com/nerd-shark/AI-Architecture-Patterns](https://github.com/nerd-shark/AI-Architecture-Patterns)

Each directory is a standalone Python project for one pattern in the series.
The goal is production-ready starting points, not tutorial scaffolding.

## Implementations

| Part | Pattern | Directory | Article |
|------|---------|-----------|---------|
| 2 | Retrieval-Augmented Generation | [p2-retrieval-augmented-generation](./p2-retrieval-augmented-generation) | *(link when published)* |

More patterns added as the series progresses.

## Design Philosophy

The article code explains *what* each pattern does. This code implements *how* to do it cleanly:

- **Abstract interfaces** at every swap point: chunker, embedder, vector store, generator. Replace any piece without touching the others.
- **Configuration as a first-class concern.** No hardcoded models, keys, thresholds, or chunk sizes. Everything flows from a config object.
- **Tests that run without API keys.** Mock implementations let you verify pipeline logic, threshold behavior, and source propagation locally.
- **Examples that show real usage.** Not "hello world" — examples that reflect the decisions you'd make in production.

## Common Setup

Python 3.11+ required. Each pattern is a separate project with its own dependencies.

```bash
cd p2-retrieval-augmented-generation
pip install -e ".[dev]"
pytest
```

See each directory's README for pattern-specific setup, configuration, and notes on swapping components for production use.
