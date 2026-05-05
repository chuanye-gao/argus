"""End-to-end PDF chunking benchmark pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import DEFAULT_STRATEGIES, chunk_document
from .evaluation import generate_evaluation_samples
from .models import ChunkingStrategy, ParsedDocument
from .pdf import parse_pdf, parse_text_file
from .report import build_recommendation_report
from .retrieval import evaluate_strategy


class BenchmarkError(ValueError):
    """Raised when a benchmark cannot be completed."""


def benchmark_pdf(
    path: str | Path,
    *,
    max_samples: int = 12,
    top_k: int = 5,
    strategies: tuple[ChunkingStrategy, ...] = DEFAULT_STRATEGIES,
) -> dict[str, Any]:
    pdf_path = Path(path)
    document = parse_pdf(pdf_path)
    return benchmark_document(
        document,
        max_samples=max_samples,
        top_k=top_k,
        strategies=strategies,
    )


def benchmark_text_file(
    path: str | Path,
    *,
    max_samples: int = 12,
    top_k: int = 5,
    strategies: tuple[ChunkingStrategy, ...] = DEFAULT_STRATEGIES,
) -> dict[str, Any]:
    document = parse_text_file(path)
    return benchmark_document(
        document,
        max_samples=max_samples,
        top_k=top_k,
        strategies=strategies,
    )


def benchmark_document(
    document: ParsedDocument,
    *,
    max_samples: int = 12,
    top_k: int = 5,
    strategies: tuple[ChunkingStrategy, ...] = DEFAULT_STRATEGIES,
) -> dict[str, Any]:
    samples = generate_evaluation_samples(document, max_samples=max_samples)
    if not samples:
        raise BenchmarkError("Could not generate evaluation samples from document text")

    strategy_list = list(strategies)
    results = []
    for strategy in strategy_list:
        chunks = chunk_document(document, strategy)
        if not chunks:
            continue
        results.append(evaluate_strategy(strategy.name, chunks, samples, top_k=top_k))

    if not results:
        raise BenchmarkError("No chunking strategy produced chunks")

    return build_recommendation_report(
        source_doc=document.source_doc,
        samples=samples,
        strategies=strategy_list,
        results=results,
    )
