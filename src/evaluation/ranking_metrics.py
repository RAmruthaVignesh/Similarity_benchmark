"""Ranking metrics for signal-discovery outputs evaluated against graded qrels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

K_VALUES = (1, 3, 5, 10)


def ranking_metrics(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    relevant_grade: int = 2,
) -> dict[str, float]:
    """Calculate graded nDCG and binary precision/recall at standard cutoffs."""
    ranked_grades = [grades.get(identifier, 0) for identifier in ranked_ids]
    ideal_grades = sorted(grades.values(), reverse=True)
    result: dict[str, float] = {}
    for k in K_VALUES:
        result[f"ndcg@{k}"] = _ndcg(ranked_grades, ideal_grades, k)
        result[f"recall@{k}"] = _recall(ranked_grades, ideal_grades, k, relevant_grade)
        result[f"precision@{k}"] = _precision(ranked_grades, k, relevant_grade)
    return result


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], 1)
    )


def _ndcg(ranked: Sequence[int], ideal: Sequence[int], k: int) -> float:
    denominator = _dcg(ideal, k)
    return 0.0 if denominator == 0 else _dcg(ranked, k) / denominator


def _recall(
    ranked: Sequence[int], all_grades: Sequence[int], k: int, threshold: int
) -> float:
    relevant = sum(grade >= threshold for grade in all_grades)
    return (
        0.0
        if relevant == 0
        else sum(grade >= threshold for grade in ranked[:k]) / relevant
    )


def _precision(ranked: Sequence[int], k: int, threshold: int) -> float:
    retrieved = ranked[:k]
    return (
        0.0
        if not retrieved
        else sum(grade >= threshold for grade in retrieved) / len(retrieved)
    )
