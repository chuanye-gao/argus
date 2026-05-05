"""Small dependency-free lexical retrieval and metric evaluation."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

from .mapping import map_evidence_to_chunks
from .models import Chunk, EvaluationSample


@dataclass(frozen=True)
class QueryResult:
    query: str
    query_type: str
    difficulty: str
    gold_chunk_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[str, ...]
    latency_ms: float

    def recall_at(self, k: int) -> float:
        if not self.gold_chunk_ids:
            return 0.0
        retrieved = set(self.retrieved_chunk_ids[:k])
        return len(retrieved.intersection(self.gold_chunk_ids)) / len(self.gold_chunk_ids)

    def reciprocal_rank_at(self, k: int) -> float:
        gold = set(self.gold_chunk_ids)
        for rank, chunk_id in enumerate(self.retrieved_chunk_ids[:k], 1):
            if chunk_id in gold:
                return 1.0 / rank
        return 0.0

    def ndcg_at(self, k: int) -> float:
        gold = set(self.gold_chunk_ids)
        dcg = 0.0
        for rank, chunk_id in enumerate(self.retrieved_chunk_ids[:k], 1):
            if chunk_id in gold:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(gold), k)
        if ideal_hits == 0:
            return 0.0
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        return dcg / ideal


@dataclass(frozen=True)
class StrategyResult:
    strategy: str
    query_results: tuple[QueryResult, ...]
    chunk_count: int
    avg_chunk_length: float
    avg_retrieved_context_length: float
    estimated_embedding_token_cost: int

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "strategy": self.strategy,
            "recall_at_1": _mean(result.recall_at(1) for result in self.query_results),
            "recall_at_3": _mean(result.recall_at(3) for result in self.query_results),
            "recall_at_5": _mean(result.recall_at(5) for result in self.query_results),
            "mrr_at_5": _mean(result.reciprocal_rank_at(5) for result in self.query_results),
            "ndcg_at_5": _mean(result.ndcg_at(5) for result in self.query_results),
            "chunk_count": self.chunk_count,
            "avg_chunk_length": round(self.avg_chunk_length, 2),
            "avg_retrieved_context_length": round(self.avg_retrieved_context_length, 2),
            "estimated_embedding_token_cost": self.estimated_embedding_token_cost,
            "avg_latency_ms": round(_mean(result.latency_ms for result in self.query_results), 3),
        }


class LexicalIndex:
    """A tiny cosine-similarity index over token counters."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._vectors = [_vectorize(chunk.text) for chunk in chunks]
        self._norms = [_norm(vector) for vector in self._vectors]

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        query_vector = _vectorize(query)
        query_norm = _norm(query_vector)
        scored: list[tuple[float, int, Chunk]] = []
        for index, chunk in enumerate(self._chunks):
            score = _cosine(query_vector, query_norm, self._vectors[index], self._norms[index])
            scored.append((score, -chunk.length, chunk))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [chunk for _, _, chunk in scored[:top_k]]


def evaluate_strategy(
    strategy_name: str,
    chunks: list[Chunk],
    samples: list[EvaluationSample],
    top_k: int = 5,
) -> StrategyResult:
    index = LexicalIndex(chunks)
    query_results: list[QueryResult] = []
    retrieved_lengths: list[int] = []

    for sample in samples:
        gold = tuple(map_evidence_to_chunks(sample, chunks))
        start = time.perf_counter()
        retrieved = index.search(sample.query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        retrieved_lengths.append(sum(chunk.length for chunk in retrieved))
        query_results.append(
            QueryResult(
                query=sample.query,
                query_type=sample.query_type,
                difficulty=sample.difficulty,
                gold_chunk_ids=gold,
                retrieved_chunk_ids=tuple(chunk.chunk_id for chunk in retrieved),
                latency_ms=latency_ms,
            )
        )

    return StrategyResult(
        strategy=strategy_name,
        query_results=tuple(query_results),
        chunk_count=len(chunks),
        avg_chunk_length=_mean(chunk.length for chunk in chunks),
        avg_retrieved_context_length=_mean(retrieved_lengths),
        estimated_embedding_token_cost=sum(_estimate_tokens(chunk.text) for chunk in chunks),
    )


def _vectorize(text: str) -> Counter[str]:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*|\d+", text)]
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.extend(cjk)
    return Counter(tokens)


def _norm(vector: Counter[str]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def _cosine(
    left: Counter[str],
    left_norm: float,
    right: Counter[str],
    right_norm: float,
) -> float:
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    return dot / (left_norm * right_norm)


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _mean(values) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
