"""Path-exact and provenance-soft metrics for controlled signal rankings."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def provenance_key(candidate: Mapping[str, object]) -> tuple[object, ...]:
    """Collapse source/path variants to the underlying attribute or FK relation.

    Attribute identity is its physical table and column. Relationship identity
    is the set of undirected FK edges traversed, so reverse traversal is treated
    as the same relational evidence. This is a diagnostic view, not a claim
    that different paths are equivalent for every downstream task.
    """
    kind = str(candidate["kind"])
    if kind == "attribute":
        return ("attribute", candidate.get("attribute_table"), candidate.get("attribute_column"))
    if kind == "relationship":
        path = tuple(str(table) for table in candidate.get("path", ()))
        edges = tuple(sorted(tuple(sorted((left, right))) for left, right in zip(path, path[1:])))
        return ("relationship", edges)
    raise ValueError(f"unknown signal kind: {kind!r}")


def ranking_metrics(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    candidates: Mapping[str, Mapping[str, object]],
    *,
    ks: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """Return path-exact and provenance-soft nDCG/recall, plus MRR@5.

    Soft qrels take the maximum grade among path variants with the same
    provenance key. Repeated predictions of the same key receive credit only
    at their first occurrence. Recall counts grade-3 signals (exact IDs or
    unique provenance keys, respectively).
    """
    missing = (set(ranked_ids) | set(grades)) - set(candidates)
    if missing:
        raise ValueError(f"candidate metadata missing for signal IDs: {sorted(missing)[:3]}")

    exact_ranked = [int(grades.get(signal_id, 0)) for signal_id in ranked_ids]
    exact_ideal = [int(grade) for grade in grades.values()]
    soft_grade_by_key: dict[tuple[object, ...], int] = {}
    for signal_id, grade in grades.items():
        key = provenance_key(candidates[signal_id])
        soft_grade_by_key[key] = max(soft_grade_by_key.get(key, 0), int(grade))

    seen: set[tuple[object, ...]] = set()
    soft_ranked: list[int] = []
    for signal_id in ranked_ids:
        key = provenance_key(candidates[signal_id])
        soft_ranked.append(soft_grade_by_key.get(key, 0) if key not in seen else 0)
        seen.add(key)
    soft_ideal = list(soft_grade_by_key.values())

    result: dict[str, float] = {}
    for k in ks:
        result[f"path_exact_ndcg@{k}"] = _ndcg(exact_ranked, k, exact_ideal)
        result[f"provenance_soft_ndcg@{k}"] = _ndcg(soft_ranked, k, soft_ideal)
        result[f"path_exact_recall@{k}"] = _recall(exact_ranked, exact_ideal, k)
        result[f"provenance_soft_recall@{k}"] = _recall(soft_ranked, soft_ideal, k)
    result["path_exact_mrr@5"] = _mrr(exact_ranked[:5])
    result["provenance_soft_mrr@5"] = _mrr(soft_ranked[:5])
    return result


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def _ndcg(ranked: Sequence[int], k: int, ideal: Sequence[int]) -> float:
    denominator = _dcg(sorted(ideal, reverse=True), k)
    return 0.0 if denominator == 0 else _dcg(ranked, k) / denominator


def _recall(ranked: Sequence[int], all_grades: Sequence[int], k: int) -> float:
    relevant = sum(grade == 3 for grade in all_grades)
    return 0.0 if relevant == 0 else sum(grade == 3 for grade in ranked[:k]) / relevant


def _mrr(grades: Sequence[int]) -> float:
    for rank, grade in enumerate(grades, 1):
        if grade == 3:
            return 1.0 / rank
    return 0.0
