#!/usr/bin/env python3
"""Evaluate schema-signal rankers on controlled real-RelAmazon-schema tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import SignalCandidate, enumerate_signal_candidates  # noqa: E402
from src.signals.scorers import (  # noqa: E402
    DeciderSignalScorer,
    MiniLMCrossEncoderScorer,
    Qwen3RerankerScorer,
    ScoredSignal,
)
from src.synthetic.controlled_signal_benchmark_v2 import (  # noqa: E402
    BENCHMARK_VERSION,
    EXPECTED_CANDIDATE_COUNT,
    MAX_HOPS,
    ControlledSignalTask,
    load_tasks,
)

DEFAULT_ROOT = REPO_ROOT / "data" / "synthetic" / "controlled-rel-amazon-signals-v2"
K_VALUES = (1, 3, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--scorer",
        action="append",
        choices=("minilm", "decider", "qwen3-reranker", "random"),
        required=True,
        help="repeat to compare several methods",
    )
    parser.add_argument("--model-name", help="optional model ID; valid with one non-random scorer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="random-baseline seed")
    parser.add_argument("--output", type=Path, help="summary JSON path")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    scorer_names = tuple(dict.fromkeys(args.scorer))
    if args.model_name and len(scorer_names) != 1:
        parser.error("--model-name requires exactly one scorer")
    if args.model_name and scorer_names[0] == "random":
        parser.error("--model-name cannot be used with random")

    try:
        dataset = LocalRelBenchDataset(args.benchmark_root)
        graph = SchemaGraph.from_manifest(dataset.schema)
        tasks = load_tasks(args.benchmark_root)
        candidates = enumerate_signal_candidates(dataset, graph, max_hops=MAX_HOPS)
        metadata = json.loads((args.benchmark_root / "benchmark.json").read_text())
        if metadata.get("benchmark_version") != BENCHMARK_VERSION:
            raise ValueError(
                f"expected benchmark {BENCHMARK_VERSION}, got "
                f"{metadata.get('benchmark_version')!r}"
            )
        if len(candidates) != EXPECTED_CANDIDATE_COUNT:
            raise ValueError(
                f"expected {EXPECTED_CANDIDATE_COUNT} candidates, got {len(candidates)}"
            )
        candidate_fingerprint = hashlib.sha256(
            "\n".join(sorted(candidate.id for candidate in candidates)).encode("utf-8")
        ).hexdigest()
        if metadata.get("candidate_id_sha256") != candidate_fingerprint:
            raise ValueError(
                "benchmark candidate IDs differ from the saved candidate catalog; "
                "rebuild the controlled benchmark"
            )
        rows = []
        for scorer_name in scorer_names:
            scorer = _build_scorer(scorer_name, args)
            for task in tasks:
                _validate_task(task, candidates)
                ranked = _rank(scorer_name, scorer, task, candidates, graph, args.seed)
                rows.append(_score_task(scorer_name, task, ranked, graph))
            scorer = None
            _release_scorer()
        summary = _summarize(rows)
        summary["candidate_count"] = len(candidates)
        summary["candidate_id_sha256"] = candidate_fingerprint
        summary["max_hops"] = MAX_HOPS
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    _write_results(summary, rows, args.output)
    for scorer_name, scores in summary["scorers"].items():
        print(
            f"{scorer_name}: nDCG@3={scores['ndcg@3']:.3f} "
            f"Recall@3={scores['recall@3']:.3f} MRR={scores['mrr']:.3f} "
            f"({scores['task_count']} tasks)"
        )
    return 0


def _build_scorer(name: str, args: argparse.Namespace) -> Any | None:
    if name == "random":
        return None
    if name == "minilm":
        return MiniLMCrossEncoderScorer(
            args.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=args.device,
            batch_size=args.batch_size,
        )
    if name == "qwen3-reranker":
        return Qwen3RerankerScorer(
            args.model_name or "Qwen/Qwen3-Reranker-4B",
            device=args.device,
            batch_size=args.batch_size,
        )
    return DeciderSignalScorer(args.model_name or "Mapika/decider-4b")


def _rank(
    name: str,
    scorer: Any | None,
    task: ControlledSignalTask,
    candidates: Sequence[SignalCandidate],
    graph: SchemaGraph,
    seed: int,
) -> Sequence[ScoredSignal]:
    if name != "random":
        assert scorer is not None
        return scorer.score(task.condition, candidates, graph)
    rng = random.Random(_stable_seed(seed, task.id))
    scored = [ScoredSignal(candidate, rng.random()) for candidate in candidates]
    return tuple(sorted(scored, key=lambda item: (-item.score, item.candidate.id)))


def _stable_seed(seed: int, task_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _release_scorer() -> None:
    """Release a model before the next backend is loaded in a combined run."""
    gc.collect()
    try:  # Torch is optional for the random and Decider-only configurations.
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _validate_task(task: ControlledSignalTask, candidates: Sequence[SignalCandidate]) -> None:
    candidate_ids = {candidate.id for candidate in candidates}
    if set(task.signal_grades) != candidate_ids:
        raise ValueError(
            f"task {task.id!r} no longer matches the standard full candidate pool; "
            "rebuild the controlled benchmark"
        )
    if not any(grade == 3 for grade in task.signal_grades.values()):
        raise ValueError(f"task {task.id!r} has no central grade-3 signal")


def _score_task(
    scorer_name: str,
    task: ControlledSignalTask,
    ranked: Sequence[ScoredSignal],
    graph: SchemaGraph,
) -> dict[str, object]:
    grades = [task.signal_grades[item.candidate.id] for item in ranked]
    row: dict[str, object] = {
        "scorer": scorer_name,
        "task_id": task.id,
        "condition": task.condition,
        "source_table": task.source_table,
        "reasoning_type": task.reasoning_type,
        "candidate_count": len(ranked),
        "top_signals": [
            {
                "rank": index,
                "id": item.candidate.id,
                "description": item.candidate.describe(graph),
                "score": item.score,
                "grade": grades[index - 1],
            }
            for index, item in enumerate(ranked[:5], 1)
        ],
    }
    for k in K_VALUES:
        row[f"ndcg@{k}"] = _ndcg(grades, k)
        row[f"recall@{k}"] = _recall_primary(grades, task.signal_grades.values(), k)
    row["mrr"] = _mrr_primary(grades)
    return row


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def _ndcg(grades: Sequence[int], k: int) -> float:
    denominator = _dcg(sorted(grades, reverse=True), k)
    return 0.0 if denominator == 0 else _dcg(grades, k) / denominator


def _recall_primary(ranked_grades: Sequence[int], all_grades: Iterable[int], k: int) -> float:
    total = sum(grade == 3 for grade in all_grades)
    return 0.0 if total == 0 else sum(grade == 3 for grade in ranked_grades[:k]) / total


def _mrr_primary(grades: Sequence[int]) -> float:
    for rank, grade in enumerate(grades, 1):
        if grade == 3:
            return 1.0 / rank
    return 0.0


def _summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_scorer: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scorer[str(row["scorer"])].append(row)
    metric_names = ("ndcg@1", "ndcg@3", "ndcg@5", "recall@1", "recall@3", "recall@5", "mrr")
    return {
        "benchmark": BENCHMARK_VERSION,
        "scorers": {
            name: {
                "task_count": len(scorer_rows),
                **{metric: sum(float(row[metric]) for row in scorer_rows) / len(scorer_rows) for metric in metric_names},
            }
            for name, scorer_rows in by_scorer.items()
        },
    }


def _write_results(summary: dict[str, object], rows: Sequence[dict[str, object]], output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_path = output.with_name(output.stem + "-tasks.jsonl")
    with task_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote summary to {output} and task scores to {task_path}")


if __name__ == "__main__":
    raise SystemExit(main())
