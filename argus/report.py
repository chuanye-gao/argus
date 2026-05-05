"""Recommendation report assembly for chunking benchmark results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import ChunkingStrategy, EvaluationSample
from .retrieval import StrategyResult


def build_recommendation_report(
    source_doc: str,
    samples: list[EvaluationSample],
    strategies: list[ChunkingStrategy],
    results: list[StrategyResult],
) -> dict[str, Any]:
    if not results:
        raise ValueError("At least one strategy result is required")

    strategy_by_name = {strategy.name: strategy for strategy in strategies}
    ranked = sorted(results, key=_ranking_key, reverse=True)
    best = ranked[0]
    best_strategy = strategy_by_name.get(best.strategy)

    return {
        "source_doc": source_doc,
        "recommended_strategy": _strategy_dict(best_strategy, best.strategy),
        "summary": {
            "reason": _reason(best, ranked),
            "evaluation_sample_count": len(samples),
        },
        "metrics": [result.metrics() for result in ranked],
        "best_by_query_type": _best_by_group(results, "query_type"),
        "best_by_difficulty": _best_by_group(results, "difficulty"),
        "failure_analysis": _failure_analysis(best, results),
        "evaluation_samples": [sample.to_dict() for sample in samples],
    }


def _ranking_key(result: StrategyResult) -> tuple[float, float, float, float, float]:
    metrics = result.metrics()
    recall = float(metrics["recall_at_5"])
    mrr = float(metrics["mrr_at_5"])
    ndcg = float(metrics["ndcg_at_5"])
    cost_penalty = 1.0 / max(1, int(metrics["estimated_embedding_token_cost"]))
    chunk_penalty = 1.0 / max(1, int(metrics["chunk_count"]))
    return (recall, mrr, ndcg, cost_penalty, chunk_penalty)


def _strategy_dict(strategy: ChunkingStrategy | None, name: str) -> dict[str, Any]:
    if strategy is None:
        return {"name": name}
    return {
        "name": strategy.name,
        "method": strategy.method,
        "chunk_size": strategy.chunk_size,
        "overlap": strategy.overlap,
    }


def _reason(best: StrategyResult, ranked: list[StrategyResult]) -> str:
    metrics = best.metrics()
    recall = float(metrics["recall_at_5"])
    mrr = float(metrics["mrr_at_5"])
    if len(ranked) == 1:
        return f"{best.strategy} was the only evaluated strategy and reached Recall@5={recall:.3f}."
    return (
        f"{best.strategy} achieved the strongest retrieval quality "
        f"(Recall@5={recall:.3f}, MRR@5={mrr:.3f}) while keeping chunk count "
        f"at {metrics['chunk_count']}."
    )


def _best_by_group(results: list[StrategyResult], field: str) -> dict[str, str]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        for query_result in result.query_results:
            group = getattr(query_result, field)
            grouped[group][result.strategy].append(query_result.recall_at(5))

    winners: dict[str, str] = {}
    for group, strategy_scores in grouped.items():
        winners[group] = max(
            strategy_scores.items(),
            key=lambda item: (sum(item[1]) / len(item[1]), -len(item[1])),
        )[0]
    return winners


def _failure_analysis(
    best: StrategyResult,
    results: list[StrategyResult],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    by_query = {
        result.strategy: {query.query: query for query in result.query_results}
        for result in results
    }

    for query_result in best.query_results:
        failed_strategies = [
            strategy
            for strategy, query_map in by_query.items()
            if query_map[query_result.query].recall_at(5) == 0.0
        ]
        if query_result.recall_at(5) == 0.0 or failed_strategies:
            failures.append(
                {
                    "query": query_result.query,
                    "query_type": query_result.query_type,
                    "difficulty": query_result.difficulty,
                    "best_strategy_rank": _rank_for_query(best.strategy, query_result.query, results),
                    "failed_strategies": failed_strategies,
                }
            )
    return failures[:10]


def _rank_for_query(strategy: str, query: str, results: list[StrategyResult]) -> int:
    ranked = sorted(
        (
            (result.strategy, _query_recall(result, query), _query_mrr(result, query))
            for result in results
        ),
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    for index, (name, _, _) in enumerate(ranked, 1):
        if name == strategy:
            return index
    return len(ranked)


def _query_recall(result: StrategyResult, query: str) -> float:
    for query_result in result.query_results:
        if query_result.query == query:
            return query_result.recall_at(5)
    return 0.0


def _query_mrr(result: StrategyResult, query: str) -> float:
    for query_result in result.query_results:
        if query_result.query == query:
            return query_result.reciprocal_rank_at(5)
    return 0.0
