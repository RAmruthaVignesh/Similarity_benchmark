#!/usr/bin/env python3
"""Evaluate schema-signal rankers on controlled real-RelAmazon-schema tasks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import SignalCandidate, enumerate_signal_candidates  # noqa: E402
from src.signals.graph_navigation import (  # noqa: E402
    DeciderGraphNavigator,
    LLMGraphNavigator,
    format_explored_schema_view,
)
from src.signals.llm_discovery import LLMDiscoveryConfig, LLMPointwiseSignalScorer  # noqa: E402
from src.signals.model_registry import (  # noqa: E402
    DISCOVERY_METHODS,
    DISCOVERY_MODEL_PROFILES,
    resolve_discovery_model,
)
from src.signals.scorers import (  # noqa: E402
    DeciderSignalScorer,
    MiniLMCrossEncoderScorer,
    ScoredSignal,
    create_choice_decider_client,
)
from src.synthetic.controlled_signal_benchmark_v2 import (  # noqa: E402
    BENCHMARK_VERSION,
    EXPECTED_CANDIDATE_COUNT,
    MAX_HOPS,
    ControlledSignalTask,
    load_tasks,
)
from src.synthetic.controlled_signal_metrics import ranking_metrics  # noqa: E402

DEFAULT_ROOT = (
    REPO_ROOT / "benchmarks" / "controlled-signal-discovery" / "rel-amazon-v2"
)
K_VALUES = (1, 3, 5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--method",
        choices=DISCOVERY_METHODS,
        required=True,
        help=(
            "score the full candidate catalog, or graph-search and rerank its shortlist; "
            "the oracle-root ablation uses each controlled task's known source table"
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(DISCOVERY_MODEL_PROFILES),
        required=True,
        help="validated model profile used to execute the method",
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--model-cache-root", type=Path, help="shared Hugging Face cache root")
    parser.add_argument("--max-tokens", type=int, default=256, help="LLM JSON generation cap")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--oci-compartment-id", default=os.environ.get("OCI_GENAI_COMPARTMENT"))
    parser.add_argument("--oci-config", type=Path, default=Path("~/.oci/config"))
    parser.add_argument("--oci-profile", default="DEFAULT")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="random-baseline seed")
    parser.add_argument("--max-decisions", type=int, default=30)
    parser.add_argument("--branch-width", type=int, default=3)
    parser.add_argument("--max-signals", type=int, default=10)
    parser.add_argument("--output", type=Path, help="summary JSON path")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    profile = resolve_discovery_model(args.model)
    is_graph = args.method in {"graph-search", "graph-search-oracle-root"}
    oracle_root = args.method == "graph-search-oracle-root"
    is_llm = profile.runner == "llm"
    if args.method not in profile.supported_methods:
        parser.error(f"--model {profile.name} does not support --method {args.method}")
    if oracle_root and args.max_decisions < 2:
        parser.error("oracle-root evaluation requires --max-decisions of at least 2")

    try:
        if is_llm and args.model_cache_root:
            _configure_model_cache(args.model_cache_root)
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
        if is_graph:
            if is_llm:
                scorer = LLMPointwiseSignalScorer(_llm_config(args, profile))
                navigator = LLMGraphNavigator(
                    scorer,
                    max_hops=MAX_HOPS,
                    max_decisions=args.max_decisions,
                    branch_width=args.branch_width,
                    max_signals=args.max_signals,
                )
            else:
                backend = _decider_backend(profile.runner)
                assert profile.model_id is not None
                shared_decider = create_choice_decider_client(
                    backend, profile.model_id, device=args.device
                )
                scorer = DeciderSignalScorer(
                    profile.model_id, backend=backend, device=args.device, client=shared_decider
                )
                navigator = DeciderGraphNavigator(
                    model_name=profile.model_id,
                    backend=backend,
                    device=args.device,
                    max_hops=MAX_HOPS,
                    max_decisions=args.max_decisions,
                    branch_width=args.branch_width,
                    max_signals=args.max_signals,
                    client=shared_decider,
                )
        else:
            scorer = _build_scorer(profile, args)
            navigator = None
        for task in tasks:
            _validate_task(task, candidates)
            if navigator is None:
                ranked = _rank(profile.runner, scorer, task, candidates, graph, args.seed)
            else:
                search_result = navigator.discover(
                    task.condition,
                    candidates,
                    graph,
                    start_table=task.source_table if oracle_root else None,
                )
                explored_schema_context = format_explored_schema_view(
                    search_result,
                    graph,
                    forced_root=task.source_table if oracle_root else None,
                )
                ranked = scorer.score(
                    task.condition,
                    search_result.signals,
                    graph,
                    explored_schema_context=explored_schema_context,
                )
            score_row = _score_task(
                profile.name, task, ranked, graph, len(candidates), candidates
            )
            score_row.update({
                "method": args.method,
                "model": profile.name,
                "model_id": getattr(scorer, "model_name", profile.model_id),
                "runner": profile.runner,
            })
            if is_graph:
                score_row["root_mode"] = "oracle" if oracle_root else "model_selected"
                score_row["ranking_stage"] = (
                    "LLM pointwise reranking from explored schema view over discovered shortlist"
                    if is_llm else "Decider ranking from explored schema view over discovered shortlist"
                )
                score_row["root_selection"] = _root_selection_record(
                    search_result, task, oracle_root=oracle_root
                )
            rows.append(score_row)
        scorer = None
        _release_scorer()
        summary = _summarize(rows)
        summary["result_format"] = "controlled-signal-discovery-v3"
        summary["method"] = args.method
        summary["model"] = profile.name
        summary["model_id"] = profile.model_id
        summary["runner"] = profile.runner
        summary["candidate_count"] = len(candidates)
        summary["candidate_id_sha256"] = candidate_fingerprint
        summary["max_hops"] = MAX_HOPS
        summary["metric_definitions"] = {
            "path_exact": "A prediction matches only the exact source-table/path candidate ID in the qrels.",
            "provenance_soft": (
                "Alternate paths to the same physical attribute or FK-edge set share the maximum qrel grade; "
                "duplicate provenance identities are credited only at first occurrence."
            ),
            "mrr_at_5": "Reciprocal rank of the first grade-3 signal within the saved top five; zero if absent.",
        }
        if is_graph:
            summary["search_config"] = {
                "max_decisions_including_root_choice": args.max_decisions,
                "root_mode": "oracle" if oracle_root else "model_selected",
                "root_branch_policy": "forced_single_oracle_root" if oracle_root else "branch_width_top_actions_at_root",
                "branch_width": args.branch_width,
                "max_signals": args.max_signals,
                "max_hops": MAX_HOPS,
            }
            if not oracle_root:
                root_records = [row["root_selection"] for row in rows]
                correct_count = sum(bool(record["correct"]) for record in root_records)
                summary["root_selection_accuracy"] = {
                    "task_count": len(root_records),
                    "correct_count": correct_count,
                    "accuracy": correct_count / len(root_records) if root_records else 0.0,
                    "definition": "selected initial table equals the controlled task source_table",
                }
                width_hits = sum(bool(record["explored_correct"]) for record in root_records)
                summary["root_selection_hit_at_width"] = {
                    "task_count": len(root_records),
                    "hit_count": width_hits,
                    "hit_rate": width_hits / len(root_records) if root_records else 0.0,
                    "width": args.branch_width,
                    "definition": "authored source_table is among the distinct initial tables explored",
                }
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    _write_results(summary, rows, args.output)
    for model_name, scores in summary["models"].items():
        print(
            f"{model_name}: nDCG@3={scores['ndcg@3']:.3f} "
            f"Recall@3={scores['recall@3']:.3f} MRR={scores['mrr']:.3f} "
            f"({scores['task_count']} tasks)"
        )
    if "root_selection_accuracy" in summary:
        top1 = summary["root_selection_accuracy"]
        hit_width = summary["root_selection_hit_at_width"]
        print(
            f"root table: top-1 exact={top1['correct_count']}/{top1['task_count']} "
            f"({top1['accuracy']:.1%}); expected table explored at width "
            f"{hit_width['width']}={hit_width['hit_count']}/{hit_width['task_count']} "
            f"({hit_width['hit_rate']:.1%})"
        )
    return 0


def _build_scorer(profile: Any, args: argparse.Namespace) -> Any | None:
    if profile.runner == "random":
        return None
    if profile.runner == "minilm":
        assert profile.model_id is not None
        return MiniLMCrossEncoderScorer(
            profile.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    if profile.runner == "strands-decider":
        assert profile.model_id is not None
        return DeciderSignalScorer(
            profile.model_id,
            backend="strands",
            device=args.device,
        )
    if profile.runner == "llm":
        return LLMPointwiseSignalScorer(_llm_config(args, profile))
    raise ValueError(f"unsupported model runner: {profile.runner}")


def _decider_backend(runner: str) -> str:
    if runner == "strands-decider":
        return "strands"
    raise ValueError(f"{runner!r} is not a typed-choice Decider runner")


def _llm_config(args: argparse.Namespace, profile: Any) -> LLMDiscoveryConfig:
    assert profile.llm_family is not None
    assert profile.model_id is not None
    return LLMDiscoveryConfig(
        model_family=profile.llm_family,
        model_name=profile.model_id,
        model_revision=args.model_revision,
        seed=args.seed,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        oci_compartment_id=args.oci_compartment_id,
        oci_config=str(args.oci_config.expanduser()),
        oci_profile=args.oci_profile,
    )


def _configure_model_cache(cache_root: Path) -> None:
    cache_root = cache_root.expanduser().resolve()
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")


def _rank(
    runner: str,
    scorer: Any | None,
    task: ControlledSignalTask,
    candidates: Sequence[SignalCandidate],
    graph: SchemaGraph,
    seed: int,
) -> Sequence[ScoredSignal]:
    if runner != "random":
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
    model_name: str,
    task: ControlledSignalTask,
    ranked: Sequence[ScoredSignal],
    graph: SchemaGraph,
    candidate_count: int,
    candidates: Sequence[SignalCandidate] = (),
) -> dict[str, object]:
    grades = [task.signal_grades[item.candidate.id] for item in ranked]
    row: dict[str, object] = {
        "result_format": "controlled-signal-discovery-v3",
        "model": model_name,
        "task_id": task.id,
        "condition": task.condition,
        "source_table": task.source_table,
        "reasoning_type": task.reasoning_type,
        "candidate_count": candidate_count,
        "returned_signal_count": len(ranked),
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
        row[f"ndcg@{k}"] = _ndcg(grades, k, ideal_grades=task.signal_grades.values())
        row[f"recall@{k}"] = _recall_primary(grades, task.signal_grades.values(), k)
    row["mrr"] = _mrr_primary(grades)
    if candidates:
        metadata = {candidate.id: candidate.to_dict(graph) for candidate in candidates}
        dual_metrics = ranking_metrics(
            [item.candidate.id for item in ranked], task.signal_grades, metadata, ks=K_VALUES
        )
        row.update(dual_metrics)
        # Explicit aliases make the existing path-sensitive metrics easy to
        # compare with provenance-collapsed metrics in reports.
        for k in K_VALUES:
            row[f"path_exact_ndcg@{k}"] = row[f"ndcg@{k}"]
            row[f"path_exact_recall@{k}"] = row[f"recall@{k}"]
        row["path_exact_mrr"] = row["mrr"]
    return row


def _root_selection_record(
    search_result: Any,
    task: ControlledSignalTask,
    *,
    oracle_root: bool,
) -> dict[str, object]:
    if oracle_root:
        return {
            "mode": "oracle",
            "expected_table": task.source_table,
            "selected_table": task.source_table,
            "choice_probability": None,
            "correct": True,
        }

    root_event = next(
        event for event in search_result.trace if event.get("current_path") is None
    )
    selected_action = next(
        (
            action
            for action in root_event["expanded_actions"]
            if action.get("kind") == "start"
        ),
        None,
    )
    selected_table = None if selected_action is None else selected_action.get("target_table")
    explored_tables = [
        str(action.get("target_table"))
        for action in root_event["expanded_actions"]
        if action.get("kind") == "start" and action.get("target_table") is not None
    ]
    return {
        "mode": "model_selected",
        "expected_table": task.source_table,
        "selected_table": selected_table,
        "choice_probability": (
            None if selected_action is None else selected_action.get("probability")
        ),
        "correct": selected_table == task.source_table,
        "explored_tables": explored_tables,
        "explored_correct": task.source_table in explored_tables,
        "model_choice": root_event.get("model_choice"),
        "table_options": [
            {
                "table": action.get("target_table"),
                "probability": action.get("probability"),
            }
            for action in root_event["legal_actions"]
            if action.get("kind") == "start"
        ],
    }


def _dcg(grades: Sequence[int], k: int) -> float:
    return sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def _ndcg(
    grades: Sequence[int], k: int, *, ideal_grades: Iterable[int] | None = None
) -> float:
    denominator = _dcg(sorted(grades if ideal_grades is None else ideal_grades, reverse=True), k)
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
    by_model: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    metric_names = (
        "ndcg@1", "ndcg@3", "ndcg@5", "recall@1", "recall@3", "recall@5", "mrr",
        "path_exact_ndcg@1", "path_exact_ndcg@3", "path_exact_ndcg@5",
        "path_exact_recall@1", "path_exact_recall@3", "path_exact_recall@5",
        "path_exact_mrr@5", "provenance_soft_ndcg@1", "provenance_soft_ndcg@3",
        "provenance_soft_ndcg@5", "provenance_soft_recall@1",
        "provenance_soft_recall@3", "provenance_soft_recall@5",
        "provenance_soft_mrr@5",
    )
    return {
        "benchmark": BENCHMARK_VERSION,
        "models": {
            name: {
                "task_count": len(model_rows),
                **{
                    metric: sum(float(row[metric]) for row in model_rows) / len(model_rows)
                    for metric in metric_names
                    if all(metric in row for row in model_rows)
                },
            }
            for name, model_rows in by_model.items()
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
