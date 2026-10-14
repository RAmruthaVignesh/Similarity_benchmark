#!/usr/bin/env python3
"""Score discovery-result rankings against frozen human or LLM signal qrels."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotate_signals import read_jsonl  # noqa: E402

K_VALUES = (1, 3, 5, 10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, required=True, help="JSONL from judge_signals.py or annotate_signals.py")
    parser.add_argument("--results", type=Path, action="append", required=True, help="discovery JSONL; repeat for each method")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="score only judged candidates and report coverage")
    parser.add_argument("--relevant-grade", type=int, default=2, choices=(1, 2, 3), help="minimum grade for Recall/Precision")
    args = parser.parse_args()
    try:
        qrels, expected = load_qrels(args.judgments)
        summaries = [
            evaluate_result_file(path, qrels, expected, args.allow_incomplete, args.relevant_grade)
            for path in args.results
        ]
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "judgments": str(args.judgments),
        "relevant_grade": args.relevant_grade,
        "allow_incomplete": args.allow_incomplete,
        "methods": summaries,
        "metric_notes": {
            "ndcg": "graded relevance over judged candidate signals; higher is better",
            "recall": f"fraction of grade >= {args.relevant_grade} signals retrieved",
            "precision": f"fraction of retrieved signals with grade >= {args.relevant_grade}",
            "unjudged": "unjudged candidates are never silently treated as grade 0",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for summary in summaries:
        metrics = summary["metrics"]
        print(
            f"{summary['method']}: conditions={summary['condition_count']}, "
            f"qrel coverage={summary['qrel_coverage']:.1%}, "
            f"nDCG@5={metrics['ndcg@5']:.4f}, Recall@5={metrics['recall@5']:.4f}"
        )
    print(f"wrote {args.output}")
    return 0


def load_qrels(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, set[str]]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"no judgments found in {path}")
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    expected: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        condition_id = row.get("condition_id")
        candidate_id = row.get("candidate_id")
        if not isinstance(condition_id, str) or not isinstance(candidate_id, str):
            raise ValueError(f"{path}: every judgment needs condition_id and candidate_id")
        expected[condition_id].add(candidate_id)
        if row.get("status") == "judged" and isinstance(row.get("grade"), int):
            qrels[condition_id][candidate_id] = int(row["grade"])
    if not qrels:
        raise ValueError(f"{path}: no completed judged grades found")
    return dict(qrels), dict(expected)


def evaluate_result_file(
    path: Path,
    qrels: Mapping[str, Mapping[str, int]],
    expected: Mapping[str, set[str]],
    allow_incomplete: bool,
    relevant_grade: int,
) -> dict[str, Any]:
    results = read_jsonl(path)
    if not results:
        raise ValueError(f"no result rows found in {path}")
    rows = []
    method_names = set()
    for result in results:
        condition_id = result.get("condition_id")
        if not isinstance(condition_id, str):
            raise ValueError(f"{path}: result row missing condition_id")
        if condition_id not in qrels:
            continue
        signals = result.get("signals")
        if not isinstance(signals, list):
            raise ValueError(f"{path}: {condition_id} has no signals list")
        judged = qrels[condition_id]
        expected_ids = expected.get(condition_id, set())
        coverage = len(judged) / len(expected_ids) if expected_ids else 0.0
        if coverage < 1.0 and not allow_incomplete:
            raise ValueError(
                f"{path}: {condition_id} has only {coverage:.1%} completed graded qrels; "
                "finish judging or pass --allow-incomplete"
            )
        ranked_ids = [str(item["id"]) for item in signals if isinstance(item, dict) and isinstance(item.get("id"), str)]
        if allow_incomplete:
            ranked_ids = [identifier for identifier in ranked_ids if identifier in judged]
        rows.append({
            "condition_id": condition_id,
            "coverage": coverage,
            "metrics": ranking_metrics(ranked_ids, judged, relevant_grade),
        })
        method_names.add(str(result.get("scorer") or result.get("method") or path.stem))
    if not rows:
        raise ValueError(f"{path}: no results overlap judged conditions")
    aggregate = {
        key: sum(row["metrics"][key] for row in rows) / len(rows)
        for key in rows[0]["metrics"]
    }
    return {
        "method": next(iter(method_names)) if len(method_names) == 1 else path.stem,
        "result_file": str(path),
        "condition_count": len(rows),
        "qrel_coverage": sum(row["coverage"] for row in rows) / len(rows),
        "metrics": aggregate,
        "per_condition": rows,
    }


def ranking_metrics(
    ranked_ids: Sequence[str], grades: Mapping[str, int], relevant_grade: int
) -> dict[str, float]:
    ranked_grades = [grades.get(identifier, 0) for identifier in ranked_ids]
    ideal = sorted(grades.values(), reverse=True)
    result: dict[str, float] = {}
    for k in K_VALUES:
        result[f"ndcg@{k}"] = ndcg(ranked_grades, ideal, k)
        result[f"recall@{k}"] = recall(ranked_grades, ideal, k, relevant_grade)
        result[f"precision@{k}"] = precision(ranked_grades, k, relevant_grade)
    return result


def ndcg(ranked: Sequence[int], ideal: Sequence[int], k: int) -> float:
    denominator = dcg(ideal, k)
    return 0.0 if denominator == 0 else dcg(ranked, k) / denominator


def dcg(grades: Sequence[int], k: int) -> float:
    return sum((2 ** grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def recall(ranked: Sequence[int], all_grades: Sequence[int], k: int, threshold: int) -> float:
    relevant = sum(grade >= threshold for grade in all_grades)
    return 0.0 if relevant == 0 else sum(grade >= threshold for grade in ranked[:k]) / relevant


def precision(ranked: Sequence[int], k: int, threshold: int) -> float:
    retrieved = ranked[:k]
    return 0.0 if not retrieved else sum(grade >= threshold for grade in retrieved) / len(retrieved)


if __name__ == "__main__":
    raise SystemExit(main())
