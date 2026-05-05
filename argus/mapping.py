"""Map source evidence spans to strategy-specific gold chunk IDs."""

from __future__ import annotations

from .models import Chunk, EvaluationSample, EvidenceSpan


def map_evidence_to_chunks(
    sample: EvaluationSample,
    chunks: list[Chunk],
    min_overlap_ratio: float = 0.5,
) -> list[str]:
    """Return chunk IDs whose ranges cover each evidence span well enough."""

    gold_ids: list[str] = []
    for span in sample.evidence_spans:
        for chunk_id in _gold_for_span(span, chunks, min_overlap_ratio):
            if chunk_id not in gold_ids:
                gold_ids.append(chunk_id)
    return gold_ids


def map_all_samples(
    samples: list[EvaluationSample],
    chunks: list[Chunk],
    min_overlap_ratio: float = 0.5,
) -> dict[str, list[str]]:
    return {
        sample.query: map_evidence_to_chunks(sample, chunks, min_overlap_ratio)
        for sample in samples
    }


def _gold_for_span(
    span: EvidenceSpan,
    chunks: list[Chunk],
    min_overlap_ratio: float,
) -> list[str]:
    matches: list[str] = []
    best_chunk_id: str | None = None
    best_overlap = 0
    span_length = max(1, span.length)

    for chunk in chunks:
        overlap = _overlap(span.start_offset, span.end_offset, chunk.start_offset, chunk.end_offset)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk_id = chunk.chunk_id
        if overlap / span_length >= min_overlap_ratio:
            matches.append(chunk.chunk_id)

    if matches:
        return matches
    if best_chunk_id:
        return [best_chunk_id]
    return []


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))
