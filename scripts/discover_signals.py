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
    if args.list_only and args.output is None:
        parser.error("--list-only requires --output")

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
        scorer = None if args.list_only else _build_scorer(args)
        records = []
        for condition in conditions:
            if scorer is None:
                signals = [candidate.to_dict(graph) for candidate in candidates]
                probability_distribution = None
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
            records.append(
                {
                    "condition_id": condition["id"],
                    "condition": condition["condition"],
                    "scorer": "schema-only" if scorer is None else args.scorer,
                    "model_name": None if scorer is None else scorer.model_name,
                    "hop_limit": args.hop_limit,
                    "source_tables": list(args.source_tables or dataset.get_tables()),
                    "candidate_count": len(candidates),
                    "score_type": (
                        "choice_probability"
                        if args.scorer == "decider" and scorer is not None
                        else "cross_encoder_score" if scorer is not None else None
                    ),
                    "choice_probabilities": probability_distribution,
                    "signals": signals,
                }
            )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    _write_jsonl(records, args.output)
    if args.output is not None:
        print(
            f"wrote {len(records)} condition result(s), each from {len(candidates)} "
            f"schema candidates, to {args.output}"
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
