"""Data models for PDF chunking benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageText:
    source_doc: str
    page: int
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class ParsedDocument:
    source_doc: str
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        return "\n".join(page.text for page in self.pages)

    @property
    def length(self) -> int:
        if not self.pages:
            return 0
        return self.pages[-1].end_offset

    def page_for_offset(self, offset: int) -> int:
        for page in self.pages:
            if page.start_offset <= offset <= page.end_offset:
                return page.page
        return self.pages[-1].page if self.pages else 1


@dataclass(frozen=True)
class EvidenceSpan:
    source_doc: str
    page: int
    start_offset: int
    end_offset: int

    @property
    def length(self) -> int:
        return max(0, self.end_offset - self.start_offset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_doc": self.source_doc,
            "page": self.page,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }


@dataclass(frozen=True)
class EvaluationSample:
    query: str
    evidence_spans: tuple[EvidenceSpan, ...]
    query_type: str
    difficulty: str
    source_doc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "query_type": self.query_type,
            "difficulty": self.difficulty,
            "source_doc": self.source_doc,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    text: str
    start_offset: int
    end_offset: int
    page_start: int
    page_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return max(0, self.end_offset - self.start_offset)


@dataclass(frozen=True)
class ChunkingStrategy:
    name: str
    method: str
    chunk_size: int | None = None
    overlap: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
        }
