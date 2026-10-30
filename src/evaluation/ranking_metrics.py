"""Ranking metrics for signal-discovery outputs evaluated against graded qrels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

K_VALUES = (1, 3, 5, 10)


def ranking_metrics(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    relevant_grade: int = 2,
    ranked_scores: Sequence[int | float | None] | None = None,
) -> dict[str, float]:
    """Calculate graded nDCG and binary precision/recall at standard cutoffs.

    When ``ranked_scores`` is supplied, identical adjacent scores are treated as
    tied. Metrics then use the expected value over every ordering inside each
    tie block. This prevents a deterministic candidate-ID tie-break from
    changing a model's reported quality at a cutoff.
    """
    if ranked_scores is not None and len(ranked_scores) != len(ranked_ids):
        raise ValueError("ranked_scores must align one-to-one with ranked_ids")
    ranked_grades = [grades.get(identifier, 0) for identifier in ranked_ids]
    ideal_grades = sorted(grades.values(), reverse=True)
    tie_blocks = _tie_blocks(ranked_grades, ranked_scores)
    result: dict[str, float] = {}
    for k in K_VALUES:
        result[f"ndcg@{k}"] = _ndcg(tie_blocks, ideal_grades, k)
        result[f"recall@{k}"] = _recall(tie_blocks, ideal_grades, k, relevant_grade)
        result[f"precision@{k}"] = _precision(tie_blocks, k, relevant_grade)
    return result


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(grades[:k], 1)
    )


def _tie_blocks(
    grades: Sequence[int], scores: Sequence[int | float | None] | None,
) -> list[tuple[int, ...]]:
    """Group consecutive candidates with the same known ranking score."""
    if scores is None:
        return [(grade,) for grade in grades]
    blocks: list[tuple[int, ...]] = []
    current: list[int] = []
    current_score: int | float | None = None
    for grade, score in zip(grades, scores, strict=True):
        known_score = isinstance(score, (int, float)) and not isinstance(score, bool)
        if current and known_score and score == current_score:
            current.append(grade)
            continue
        if current:
            blocks.append(tuple(current))
        current = [grade]
        # Unknown scores create a singleton, rather than an artificial tie.
        current_score = score if known_score else None
    if current:
        blocks.append(tuple(current))
    return blocks


def _expected_dcg(tie_blocks: Sequence[Sequence[int]], k: int) -> float:
    """Expected DCG if every order inside a score tie is equally likely."""
    total = 0.0
    position = 1
    for block in tie_blocks:
        if position > k:
            break
        block_end = position + len(block) - 1
        observed_end = min(block_end, k)
        discounts = sum(1 / math.log2(rank + 1) for rank in range(position, observed_end + 1))
        mean_gain = sum(2**grade - 1 for grade in block) / len(block)
        total += mean_gain * discounts
        position = block_end + 1
    return total


def _ndcg(tie_blocks: Sequence[Sequence[int]], ideal: Sequence[int], k: int) -> float:
    denominator = _dcg(ideal, k)
    return 0.0 if denominator == 0 else _expected_dcg(tie_blocks, k) / denominator


def _recall(
    tie_blocks: Sequence[Sequence[int]], all_grades: Sequence[int], k: int, threshold: int
) -> float:
    relevant = sum(grade >= threshold for grade in all_grades)
    return 0.0 if relevant == 0 else _expected_relevant(tie_blocks, k, threshold) / relevant


def _expected_relevant(
    tie_blocks: Sequence[Sequence[int]], k: int, threshold: int
) -> float:
    total = 0.0
    position = 1
    for block in tie_blocks:
        if position > k:
            break
        block_end = position + len(block) - 1
        selected = min(block_end, k) - position + 1
        total += selected * sum(grade >= threshold for grade in block) / len(block)
        position = block_end + 1
    return total


def _precision(tie_blocks: Sequence[Sequence[int]], k: int, threshold: int) -> float:
    retrieved = min(k, sum(len(block) for block in tie_blocks))
    return 0.0 if retrieved == 0 else _expected_relevant(tie_blocks, k, threshold) / retrieved
