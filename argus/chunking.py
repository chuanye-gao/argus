"""Chunking strategies for PDF benchmark experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Chunk, ChunkingStrategy, ParsedDocument


DEFAULT_STRATEGIES: tuple[ChunkingStrategy, ...] = tuple(
    ChunkingStrategy(
        name=f"fixed_char_{chunk_size}_{overlap}",
        method="fixed_char",
        chunk_size=chunk_size,
        overlap=overlap,
    )
    for chunk_size in (500, 800, 1200)
    for overlap in (50, 100, 200)
    if overlap < chunk_size
) + (
    ChunkingStrategy(
        name="recursive_character_800_100",
        method="recursive_character",
        chunk_size=800,
        overlap=100,
    ),
    ChunkingStrategy(
        name="paragraph_1200",
        method="paragraph",
        chunk_size=1200,
        overlap=0,
    ),
)


def chunk_document(
    document: ParsedDocument,
    strategy: ChunkingStrategy,
) -> list[Chunk]:
    if strategy.method == "fixed_char":
        return _fixed_char_chunks(document, strategy)
    if strategy.method == "recursive_character":
        return _recursive_chunks(document, strategy)
    if strategy.method == "paragraph":
        return _paragraph_chunks(document, strategy)
    raise ValueError(f"Unknown chunking method: {strategy.method}")


def chunk_all(
    document: ParsedDocument,
    strategies: Iterable[ChunkingStrategy] = DEFAULT_STRATEGIES,
) -> dict[str, list[Chunk]]:
    return {strategy.name: chunk_document(document, strategy) for strategy in strategies}


def _fixed_char_chunks(document: ParsedDocument, strategy: ChunkingStrategy) -> list[Chunk]:
    chunk_size = _require_chunk_size(strategy)
    overlap = max(0, strategy.overlap)
    step = max(1, chunk_size - overlap)
    text = document.text
    chunks: list[Chunk] = []

    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_make_chunk(document, strategy, len(chunks), start, end, chunk_text))
        if end == len(text):
            break
        start += step

    return chunks


def _recursive_chunks(document: ParsedDocument, strategy: ChunkingStrategy) -> list[Chunk]:
    chunk_size = _require_chunk_size(strategy)
    overlap = max(0, strategy.overlap)
    ranges = _split_by_boundaries(document.text)
    return _pack_ranges(document, strategy, ranges, chunk_size, overlap)


def _paragraph_chunks(document: ParsedDocument, strategy: ChunkingStrategy) -> list[Chunk]:
    chunk_size = strategy.chunk_size or 1200
    ranges = _paragraph_ranges(document.text)
    return _pack_ranges(document, strategy, ranges, chunk_size, 0)


def _split_by_boundaries(text: str) -> list[tuple[int, int]]:
    paragraphs = _paragraph_ranges(text)
    ranges: list[tuple[int, int]] = []
    for start, end in paragraphs:
        paragraph = text[start:end]
        if len(paragraph) <= 450:
            ranges.append((start, end))
            continue
        cursor = start
        for match in re.finditer(r"[^.!?。！？]+[.!?。！？]?", paragraph):
            sentence = match.group(0)
            if sentence.strip():
                ranges.append((cursor + match.start(), cursor + match.end()))
    return ranges or [(0, len(text))]


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S):
        start, end = match.span()
        if text[start:end].strip():
            ranges.append((start, end))
    if ranges:
        return ranges

    for match in re.finditer(r"[^\n]+", text):
        start, end = match.span()
        if text[start:end].strip():
            ranges.append((start, end))
    return ranges


def _pack_ranges(
    document: ParsedDocument,
    strategy: ChunkingStrategy,
    ranges: list[tuple[int, int]],
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    current_start: int | None = None
    current_end: int | None = None

    for start, end in ranges:
        if current_start is None:
            current_start, current_end = start, end
            continue

        assert current_end is not None
        if end - current_start <= chunk_size:
            current_end = end
            continue

        _append_window(document, strategy, chunks, current_start, current_end)
        current_start = _overlap_start(document.text, current_start, current_end, overlap)
        current_end = end

    if current_start is not None and current_end is not None:
        _append_window(document, strategy, chunks, current_start, current_end)

    return chunks


def _append_window(
    document: ParsedDocument,
    strategy: ChunkingStrategy,
    chunks: list[Chunk],
    start: int,
    end: int,
) -> None:
    text = document.text[start:end].strip()
    if text:
        chunks.append(_make_chunk(document, strategy, len(chunks), start, end, text))


def _overlap_start(text: str, start: int, end: int, overlap: int) -> int:
    if overlap <= 0:
        return end
    candidate = max(start, end - overlap)
    boundary = text.find(" ", candidate, end)
    if boundary >= 0:
        return boundary + 1
    return candidate


def _make_chunk(
    document: ParsedDocument,
    strategy: ChunkingStrategy,
    index: int,
    start: int,
    end: int,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=f"{strategy.name}_{index:04d}",
        strategy=strategy.name,
        text=text,
        start_offset=start,
        end_offset=end,
        page_start=document.page_for_offset(start),
        page_end=document.page_for_offset(max(start, end - 1)),
        metadata=strategy.to_dict(),
    )


def _require_chunk_size(strategy: ChunkingStrategy) -> int:
    if not strategy.chunk_size or strategy.chunk_size <= 0:
        raise ValueError(f"Strategy {strategy.name} requires a positive chunk_size")
    return strategy.chunk_size
