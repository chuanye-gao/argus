"""Strategy-independent evaluation sample generation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .chunking import _paragraph_ranges
from .models import EvaluationSample, EvidenceSpan, ParsedDocument


SINGLE_HOP = "single_hop"
MULTI_SPAN_SAME_DOC = "multi_span_same_doc"

_STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "before",
    "between",
    "can",
    "could",
    "document",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "may",
    "must",
    "not",
    "pdf",
    "should",
    "that",
    "the",
    "their",
    "there",
    "this",
    "through",
    "under",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def generate_evaluation_samples(
    document: ParsedDocument,
    max_samples: int = 12,
) -> list[EvaluationSample]:
    """Create query/evidence samples before any strategy-specific chunk labels."""

    candidates = _candidate_spans(document)
    if not candidates:
        return []

    single_target = min(max(3, len(candidates)), max_samples)
    singles: list[EvaluationSample] = []
    for index, span in enumerate(candidates[:single_target]):
        text = _span_text(document, span)
        topic = _topic_from_text(text)
        singles.append(
            EvaluationSample(
                query=_single_query(topic, index),
                evidence_spans=(span,),
                query_type=SINGLE_HOP,
                difficulty="easy" if index % 3 == 0 else "medium",
                source_doc=document.source_doc,
            )
        )

    remaining = max(0, max_samples - len(singles))
    multi_target = min(3, remaining, max(0, len(candidates) // 2))
    multis: list[EvaluationSample] = []
    for index in range(multi_target):
        first = candidates[index]
        second = candidates[-(index + 1)]
        if first == second:
            continue
        topics = _unique(
            [_topic_from_text(_span_text(document, first)), _topic_from_text(_span_text(document, second))]
        )
        multis.append(
            EvaluationSample(
                query=_multi_query(topics),
                evidence_spans=(first, second),
                query_type=MULTI_SPAN_SAME_DOC,
                difficulty="hard",
                source_doc=document.source_doc,
            )
        )

    samples = singles + multis
    _validate_samples(samples)
    return samples


def _candidate_spans(document: ParsedDocument) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    full_text = document.text
    for start, end in _paragraph_ranges(full_text):
        paragraph = full_text[start:end].strip()
        if len(paragraph) < 80:
            continue
        if len(paragraph) > 900:
            end = start + 900
        spans.append(
            EvidenceSpan(
                source_doc=document.source_doc,
                page=document.page_for_offset(start),
                start_offset=start,
                end_offset=end,
            )
        )

    if spans:
        return spans

    for match in re.finditer(r"[^.!?。！？]{80,500}[.!?。！？]?", full_text):
        start, end = match.span()
        spans.append(
            EvidenceSpan(
                source_doc=document.source_doc,
                page=document.page_for_offset(start),
                start_offset=start,
                end_offset=end,
            )
        )
    return spans


def _span_text(document: ParsedDocument, span: EvidenceSpan) -> str:
    return document.text[span.start_offset : span.end_offset]


def _single_query(topic: str, index: int) -> str:
    templates = (
        "What does the document say about {topic}?",
        "Which details are relevant when reviewing {topic}?",
        "What requirements or conditions are described for {topic}?",
    )
    return templates[index % len(templates)].format(topic=topic)


def _multi_query(topics: list[str]) -> str:
    if len(topics) >= 2:
        subject = f"{topics[0]} and {topics[1]}"
    elif topics:
        subject = topics[0]
    else:
        subject = "these related topics"
    return f"What should be considered together about {subject}?"


def _topic_from_text(text: str) -> str:
    tokens = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
        if token.lower() not in _STOP_WORDS and not token.isdigit()
    ]
    if tokens:
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], tokens.index(item[0])))
        return " ".join(word for word, _ in ranked[:3])

    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    if cjk:
        return cjk[0][:12]
    return "this section"


def _unique(values: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _validate_samples(samples: list[EvaluationSample]) -> None:
    queries: set[str] = set()
    for sample in samples:
        if not sample.query.strip():
            raise ValueError("Generated query must be non-empty")
        if sample.query in queries:
            raise ValueError(f"Duplicate query generated: {sample.query}")
        queries.add(sample.query)
        if not sample.evidence_spans:
            raise ValueError("Evaluation sample must include evidence spans")
        for span in sample.evidence_spans:
            if span.length <= 0:
                raise ValueError("Evidence span must have positive length")
