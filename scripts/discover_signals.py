#!/usr/bin/env python3
"""Enumerate relational schema signals and rank them for JSONL user intents."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import CATALOG_VIEWS, enumerate_signal_candidates  # noqa: E402
from src.signals.graph_navigation import (  # noqa: E402
    DeciderGraphNavigator,
    format_explored_schema_view,
)
from src.signals.scorers import (  # noqa: E402
    DeciderSignalScorer,
    MiniLMCrossEncoderScorer,
    RandomSignalScorer,
    create_choice_decider_client,
    default_decider_model,
)

DEFAULT_CONDITIONS = REPO_ROOT / "conditions" / "rel-amazon" / "conditions-v3.jsonl"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"
DEFAULT_ORACLE_ROOT_MAP = REPO_ROOT / "conditions" / "rel-amazon" / "oracle-root-map-v3.json"


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
        choices=("minilm", "strands-decider", "random"),
        default="minilm",
        help="local scoring backend (default: minilm)",
    )
    parser.add_argument(
        "--method",
        choices=(
            "rank",
            "decider-graph-reranked",
            "decider-graph-reranked-oracle-root",
        ),
        default="rank",
        help=(
            "rank a fixed candidate list, or navigate the graph then rerank only "
            "the discovered shortlist, optionally with a forced target-entity root "
            "(default: rank)"
        ),
    )
    parser.add_argument(
        "--oracle-root-map",
        type=Path,
        default=DEFAULT_ORACLE_ROOT_MAP,
        help=f"condition-id to starting-table JSON map for oracle-root mode (default: {DEFAULT_ORACLE_ROOT_MAP})",
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
        "--catalog-view",
        choices=(*CATALOG_VIEWS, "both"),
        default="path-aware",
        help=(
            "candidate catalog: path-aware keeps provenance-specific paths; "
            "feature-collapsed keeps one direct candidate per physical attribute; "
            "both writes one output JSONL per view (default: path-aware)"
        ),
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
        help="pairwise examples per batch for MiniLM (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random baseline seed (default: 0)")
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
    if args.catalog_view == "both" and args.output is None:
        parser.error("--catalog-view both requires --output to write one file per view")
    if args.list_only and args.method != "rank":
        parser.error("--list-only cannot be combined with graph-search methods")
    is_oracle_root = args.method == "decider-graph-reranked-oracle-root"
    is_graph = args.method in {
        "decider-graph-reranked",
        "decider-graph-reranked-oracle-root",
    }
    if is_graph and args.scorer != "strands-decider":
        parser.error("Decider graph methods require --scorer strands-decider")
    if is_graph and args.source_tables:
        parser.error(
            "Decider graph methods choose their own start table; do not pass --source-table"
        )
    if args.model_name and args.scorer == "random":
        parser.error("--model-name cannot be used with --scorer random")
    if is_oracle_root and args.max_decisions < 2:
        parser.error("oracle-root graph search requires --max-decisions of at least 2")
    if args.catalog_view == "both":
        return _run_both_catalog_views(args)

    try:
        dataset = LocalRelBenchDataset(args.dataset_root)
        graph = SchemaGraph.from_manifest(dataset.schema)
        candidates = enumerate_signal_candidates(
            dataset,
            graph,
            source_tables=args.source_tables,
            max_hops=args.hop_limit,
            include_relationships=(
                args.catalog_view == "path-aware" and not args.no_relationships
            ),
            include_structural_columns=args.include_structural_columns,
            catalog_view=args.catalog_view,
        )
        conditions = list(_load_conditions(args.conditions))
        oracle_roots = (
            _load_oracle_roots(args.oracle_root_map, conditions)
            if is_oracle_root
            else {}
        )
        shared_decider = None
        if args.list_only:
            scorer = None
        elif is_graph:
            backend = _decider_backend(args.scorer)
            model_name = args.model_name or default_decider_model(backend)
            shared_decider = create_choice_decider_client(
                backend, model_name, device=args.device
            )
            scorer = DeciderSignalScorer(
                model_name, backend=backend, device=args.device, client=shared_decider
            )
        else:
            scorer = _build_scorer(args)
        model_name = getattr(scorer, "model_name", None)
        if is_graph:
            model_name = args.model_name or model_name or default_decider_model("strands")
        navigator = (
            DeciderGraphNavigator(
                model_name=model_name,
                backend=_decider_backend(args.scorer),
                device=args.device,
                max_hops=args.hop_limit,
                max_decisions=args.max_decisions,
                branch_width=args.branch_width,
                max_signals=args.max_signals,
                include_relationships=(
                    args.catalog_view == "path-aware" and not args.no_relationships
                ),
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
                oracle_root = oracle_roots.get(condition["id"])
                result = navigator.discover(
                    condition["condition"],
                    candidates,
                    graph,
                    start_table=oracle_root,
                )
                explored_schema_context = format_explored_schema_view(
                    result, graph, forced_root=oracle_root
                )
                if len(result.signals) > 1:
                    reranked = scorer.score(
                        condition["condition"],
                        result.signals,
                        graph,
                        explored_schema_context=explored_schema_context,
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
                    if args.scorer == "strands-decider"
                    else None
                )
                score_type = (
                    "choice_probability"
                    if args.scorer == "strands-decider"
                    else "random_score"
                    if args.scorer == "random"
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
                        else args.scorer
                    ),
                    "model_name": model_name,
                    "signal_catalog": args.catalog_view,
                    "catalog_definition": (
                        "path-specific attributes and relationship paths"
                        if args.catalog_view == "path-aware"
                        else "one direct candidate per physical attribute; no relationship/path candidates"
                    ),
                    "hop_limit": args.hop_limit,
                    "source_tables": list(args.source_tables or dataset.get_tables()),
                    "candidate_count": len(candidates),
                    "score_type": score_type,
                    "choice_probabilities": probability_distribution,
                    "signals": signals,
                    **(
                        {"oracle_root_table": oracle_roots[condition["id"]]}
                        if is_oracle_root
                        else {}
                    ),
                    **({"random_seed": args.seed} if args.scorer == "random" else {}),
                    **(
                        {
                            "graph_search": graph_search,
                            "graph_search_config": {
                                "max_decisions": args.max_decisions,
                                "branch_width": args.branch_width,
                                "max_signals": args.max_signals,
                                "include_relationships": (
                                    args.catalog_view == "path-aware"
                                    and not args.no_relationships
                                ),
                                "root_mode": "target_entity_proxy" if is_oracle_root else "model_selected",
                                **(
                                    {
                                        "final_ranking": "Decider over discovered shortlist",
                                        "final_prompt_context": "explored schema view; no search choices or probabilities",
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
        run_label = "schema-only" if args.list_only else f"{args.scorer}/{args.method}"
        print(
            f"wrote {len(records)} condition result(s) using {run_label} from "
            f"{len(candidates)} schema candidates, to {args.output}"
        )
    return 0


def _run_both_catalog_views(args: argparse.Namespace) -> int:
    """Run this command once per catalog view with deterministic output names.

    Keeping each catalog in a separate process gives each run a normal, fully
    self-contained JSONL artifact and avoids mixing candidate universes in one
    file. This is deliberately a convenience mode: model weights are loaded
    once per view.
    """
    assert args.output is not None  # enforced during argument validation
    original = list(sys.argv[1:])
    for catalog_view in CATALOG_VIEWS:
        output = _catalog_output_path(args.output, catalog_view)
        child_args = _replace_cli_value(original, "--catalog-view", catalog_view)
        child_args = _replace_cli_value(child_args, "--output", str(output))
        print(f"Running {catalog_view} catalog → {output}", flush=True)
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *child_args],
            check=False,
        )
        if completed.returncode:
            return completed.returncode
    return 0


def _replace_cli_value(arguments: list[str], flag: str, value: str) -> list[str]:
    """Replace a required argparse option in either common CLI spelling."""
    replaced = list(arguments)
    for index, argument in enumerate(replaced):
        if argument == flag:
            if index + 1 >= len(replaced):
                raise ValueError(f"missing value after {flag}")
            replaced[index + 1] = value
            return replaced
        if argument.startswith(f"{flag}="):
            replaced[index] = f"{flag}={value}"
            return replaced
    raise ValueError(f"{flag} was parsed but is absent from the command line")


def _catalog_output_path(output: Path, catalog_view: str) -> Path:
    suffix = output.suffix or ".jsonl"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}-{catalog_view}{suffix}")


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


def _load_oracle_roots(
    path: Path, conditions: list[dict[str, str]]
) -> dict[str, str]:
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid oracle-root JSON: {error.msg}") from error
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise ValueError(f"{path}: expected a JSON object mapping condition IDs to table names")
    missing = sorted(condition["id"] for condition in conditions if condition["id"] not in mapping)
    if missing:
        raise ValueError(f"{path}: no oracle root supplied for conditions: {missing}")
    return mapping


def _build_scorer(args: argparse.Namespace) -> Any:
    if args.scorer == "random":
        return RandomSignalScorer(seed=args.seed)
    if args.scorer == "minilm":
        return MiniLMCrossEncoderScorer(
            model_name=args.model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=args.device,
            batch_size=args.batch_size,
        )
    if args.scorer == "strands-decider":
        return DeciderSignalScorer(
            model_name=args.model_name or default_decider_model("strands"),
            backend="strands",
            device=args.device,
        )
    raise ValueError(f"unsupported scorer: {args.scorer}")


def _decider_backend(scorer_name: str) -> str:
    if scorer_name == "strands-decider":
        return "strands"
    raise ValueError(f"{scorer_name!r} is not a typed-choice Decider scorer")


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
