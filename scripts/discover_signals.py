#!/usr/bin/env python3
"""Enumerate relational schema signals and rank them for JSONL user intents."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402
from src.signals.graph_navigation import (  # noqa: E402
    DeciderGraphNavigator,
    format_search_history,
)
from src.signals.scorers import (  # noqa: E402
    DeciderSignalScorer,
    MiniLMCrossEncoderScorer,
    Qwen3RerankerScorer,
)

DEFAULT_CONDITIONS = REPO_ROOT / "conditions" / "rel-amazon" / "conditions3.jsonl"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        type=Path,
        default=DEFAULT_CONDITIONS,
        help=f"JSONL conditions file (default: {DEFAULT_CONDITIONS})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"local relational dataset directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path; omit to print the records to stdout",
    )
    parser.add_argument(
        "--scorer",
        choices=("minilm", "decider", "qwen3-reranker"),
        default="minilm",
        help="local scoring backend (default: minilm)",
    )
    parser.add_argument(
        "--method",
        choices=("rank", "decider-graph-reranked"),
        default="rank",
        help=(
            "rank a fixed candidate list, or navigate the graph then rerank only "
            "the discovered shortlist with Decider (default: rank)"
        ),
    )
    parser.add_argument(
        "--model-name",
        help="Hugging Face model ID; uses the selected scorer's documented default",
    )
    parser.add_argument(
        "--hop-limit",
        type=int,
        default=2,
        help="maximum relational path length in foreign-key hops (default: 2)",
    )
    parser.add_argument(
        "--source-table",
        action="append",
        dest="source_tables",
        help="anchor table to enumerate from; repeat to select several (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="pairwise examples per batch for MiniLM/Qwen3 reranker (default: 32)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="MiniLM device, e.g. cuda, cuda:0, cpu (default: auto)",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=12,
        help="maximum Decider calls per condition during graph exploration (default: 12)",
    )
    parser.add_argument(
        "--branch-width",
        type=int,
        default=2,
        help="highest-probability legal actions retained per Decider call (default: 2)",
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=10,
        help="maximum signals emitted per condition before final reranking (default: 10)",
    )
    parser.add_argument(
        "--include-structural-columns",
        action="store_true",
        help="also enumerate primary keys and foreign-key identifiers",
    )
    parser.add_argument(
        "--no-relationships",
        action="store_true",
        help="omit path-only relationship candidates",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="write generated candidates without loading a scoring model",
    )
    args = parser.parse_args()
    if args.hop_limit < 0:
        parser.error("--hop-limit must be non-negative")
    if args.max_decisions <= 0:
        parser.error("--max-decisions must be positive")
    if args.branch_width <= 0:
        parser.error("--branch-width must be positive")
    if args.max_signals <= 0:
        parser.error("--max-signals must be positive")
    if args.list_only and args.output is None:
        parser.error("--list-only requires --output")
    if args.list_only and args.method != "rank":
        parser.error("--list-only cannot be combined with --method decider-graph-reranked")
    is_graph = args.method == "decider-graph-reranked"
    if is_graph and args.scorer != "decider":
        parser.error("Decider graph methods require --scorer decider")
    if is_graph and args.source_tables:
        parser.error(
            "Decider graph methods choose their own start table; do not pass --source-table"
        )

    try:
        dataset = LocalRelBenchDataset(args.dataset_root)
        graph = SchemaGraph.from_manifest(dataset.schema)
        candidates = enumerate_signal_candidates(
            dataset,
            graph,
            source_tables=args.source_tables,
            max_hops=args.hop_limit,
            include_relationships=not args.no_relationships,
            include_structural_columns=args.include_structural_columns,
        )
        conditions = list(_load_conditions(args.conditions))
        shared_decider = None
        if args.list_only:
            scorer = None
        elif is_graph:
            try:
                from decider.infer import Decider
            except ImportError as error:
                raise RuntimeError(
                    "Decider graph reranking requires the decider-ai package"
                ) from error
            model_name = args.model_name or "Mapika/decider-4b"
            shared_decider = Decider(model_name)
            scorer = DeciderSignalScorer(model_name, client=shared_decider)
        else:
            scorer = _build_scorer(args)
        model_name = getattr(scorer, "model_name", None)
        if is_graph:
            model_name = args.model_name or model_name or "Mapika/decider-4b"
        navigator = (
            DeciderGraphNavigator(
                model_name=model_name,
                max_hops=args.hop_limit,
                max_decisions=args.max_decisions,
                branch_width=args.branch_width,
                max_signals=args.max_signals,
                include_relationships=not args.no_relationships,
                client=shared_decider,
            )
            if is_graph
            else None
        )
        records = []
        for condition in conditions:
            score_type = None
            if args.list_only:
                signals = [candidate.to_dict(graph) for candidate in candidates]
                probability_distribution = None
                graph_search = None
            elif navigator is not None:
                result = navigator.discover(condition["condition"], candidates, graph)
                history_context = format_search_history(result, graph)
                if len(result.signals) > 1:
                    reranked = scorer.score(
                        condition["condition"],
                        result.signals,
                        graph,
                        history_context=history_context,
                    )
                    signals = [scored.to_dict(graph) for scored in reranked]
                    probability_distribution = {
                        scored.candidate.id: scored.score for scored in reranked
                    }
                    score_type = "choice_probability_within_discovered_shortlist"
                else:
                    signals = [candidate.to_dict(graph) for candidate in result.signals]
                    probability_distribution = None
                    score_type = "no_rerank_singleton_or_empty_shortlist"
                graph_search = result.to_dict(graph)["search"]
            else:
                scored_signals = scorer.score(condition["condition"], candidates, graph)
                signals = [
                    scored.to_dict(graph) for scored in scored_signals
                ]
                probability_distribution = (
                    {
                        scored.candidate.id: scored.score
                        for scored in scored_signals
                    }
                    if args.scorer == "decider"
                    else None
                )
                score_type = (
                    "choice_probability"
                    if args.scorer == "decider"
                    else "cross_encoder_score"
                )
                graph_search = None
            records.append(
                {
                    "condition_id": condition["id"],
                    "condition": condition["condition"],
                    "method": "schema-only" if args.list_only else args.method,
                    "scorer": (
                        "schema-only"
                        if args.list_only
                        else "decider"
                        if is_graph
                        else args.scorer
                    ),
                    "model_name": model_name,
                    "hop_limit": args.hop_limit,
                    "source_tables": list(args.source_tables or dataset.get_tables()),
                    "candidate_count": len(candidates),
                    "score_type": score_type,
                    "choice_probabilities": probability_distribution,
                    "signals": signals,
                    **(
                        {
                            "graph_search": graph_search,
                            "graph_search_config": {
                                "max_decisions": args.max_decisions,
                                "branch_width": args.branch_width,
                                "max_signals": args.max_signals,
                                **(
                                    {
                                        "final_ranking": "Decider over discovered shortlist",
                                        "final_prompt_context": "graph-search history; no full schema",
                                    }
                                    if is_graph
                                    else {}
                                ),
                            },
                        }
                        if graph_search is not None
                        else {}
                    ),
                }
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    _write_jsonl(records, args.output)
    if args.output is not None:
        print(
            f"wrote {len(records)} condition result(s) using {args.method} from "
            f"{len(candidates)} schema candidates, to {args.output}"
        )
    return 0


def _load_conditions(path: Path) -> Iterable[dict[str, str]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            identifier, condition = entry.get("id"), entry.get("condition")
            if not isinstance(identifier, str) or not identifier:
                raise ValueError(f"{path}:{line_number}: missing non-empty string 'id'")
            if not isinstance(condition, str) or not condition:
                raise ValueError(
                    f"{path}:{line_number}: missing non-empty string 'condition'"
                )
            yield {"id": identifier, "condition": condition}


def _build_scorer(args: argparse.Namespace) -> Any:
    if args.scorer == "minilm":
        return MiniLMCrossEncoderScorer(
            model_name=args.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=args.device,
            batch_size=args.batch_size,
        )
    if args.scorer == "decider":
        return DeciderSignalScorer(model_name=args.model_name or "Mapika/decider-4b")
    if args.scorer == "qwen3-reranker":
        return Qwen3RerankerScorer(
            model_name=args.model_name or "Qwen/Qwen3-Reranker-4B",
            device=args.device,
            batch_size=args.batch_size,
        )
    raise ValueError(f"unsupported scorer: {args.scorer}")


def _write_jsonl(records: Iterable[dict[str, object]], output: Path | None) -> None:
    if output is None:
        for record in records:
            print(json.dumps(record, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
