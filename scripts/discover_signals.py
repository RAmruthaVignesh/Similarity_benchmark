#!/usr/bin/env python3
"""Enumerate relational schema signals and rank them for JSONL user intents."""

from __future__ import annotations

import argparse
import json
import os
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
    RandomSignalScorer,
    create_choice_decider_client,
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
        "--method",
        choices=DISCOVERY_METHODS,
        required=True,
        help=(
            "signal-discovery method: score the full catalog, search the graph, "
            "or run graph search from an oracle-provided root"
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(DISCOVERY_MODEL_PROFILES),
        required=True,
        help="validated model profile used to execute the method",
    )
    parser.add_argument(
        "--oracle-root-map",
        type=Path,
        default=DEFAULT_ORACLE_ROOT_MAP,
        help=f"condition-id to starting-table JSON map for oracle-root mode (default: {DEFAULT_ORACLE_ROOT_MAP})",
    )
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--model-cache-root", type=Path, help="shared Hugging Face cache root")
    parser.add_argument("--max-tokens", type=int, default=256, help="LLM JSON generation cap")
    parser.add_argument("--max-attempts", type=int, default=2, help="retry malformed LLM JSON this many times")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--oci-compartment-id", default=os.environ.get("OCI_GENAI_COMPARTMENT"))
    parser.add_argument("--oci-config", type=Path, default=Path("~/.oci/config"))
    parser.add_argument("--oci-profile", default="DEFAULT")
    parser.add_argument("--resume", action="store_true", help="reuse completed condition records in --output")
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
        "--batch-size", type=int, default=32,
        help="MiniLM pairwise batch size or LLM request concurrency (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random baseline seed (default: 0)")
    parser.add_argument(
        "--device",
        default="auto",
        help="local model device, e.g. cuda, cuda:0, cpu (default: auto)",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=12,
        help="maximum model decisions per condition during graph search (default: 12)",
    )
    parser.add_argument(
        "--branch-width",
        type=int,
        default=2,
        help="legal actions retained per graph-search decision (default: 2)",
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
    profile = resolve_discovery_model(args.model)
    if args.list_only and args.method != "catalog-rank":
        parser.error("--list-only requires --method catalog-rank")
    is_llm = profile.runner == "llm"
    is_oracle_root = args.method == "graph-search-oracle-root"
    is_graph = args.method in {"graph-search", "graph-search-oracle-root"}
    if args.method not in profile.supported_methods:
        parser.error(f"--model {profile.name} does not support --method {args.method}")
    if is_graph and args.source_tables:
        parser.error(
            "graph-search strategies choose their own start table; do not pass --source-table"
        )
    if is_oracle_root and args.max_decisions < 2:
        parser.error("oracle-root graph search requires --max-decisions of at least 2")
    if args.catalog_view == "both":
        return _run_both_catalog_views(args)

    try:
        if is_llm and args.model_cache_root:
            _configure_model_cache(args.model_cache_root)
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
        shared_llm = None
        if args.list_only:
            scorer = None
        elif is_llm:
            shared_llm = LLMPointwiseSignalScorer(_llm_config(args, profile))
            scorer = shared_llm
        elif is_graph:
            decider_backend = _decider_backend(profile.runner)
            model_name = profile.model_id
            assert model_name is not None
            shared_decider = create_choice_decider_client(
                decider_backend, model_name, device=args.device
            )
            scorer = DeciderSignalScorer(
                model_name, backend=decider_backend, device=args.device, client=shared_decider
            )
        else:
            scorer = _build_scorer(args, profile)
        model_name = getattr(scorer, "model_name", None)
        if profile.runner == "strands-decider" and is_graph:
            model_name = model_name or profile.model_id
        navigator = (
            _build_graph_navigator(
                args,
                scorer,
                model_name,
                shared_decider,
                profile,
                is_llm=is_llm,
            )
            if is_graph
            else None
        )
        existing = _load_existing_records(args.output) if args.resume and args.output else {}
        records = [record for record in existing.values() if record.get("method") == args.method]
        completed_conditions = {
            str(record.get("condition_id")) for record in records
            if (
                record.get("model") == profile.name
                and record.get("model_id") == model_name
                and record.get("signal_catalog") == args.catalog_view
            )
        }
        for condition in conditions:
            if condition["id"] in completed_conditions:
                continue
            score_type = None
            if args.list_only:
                signals = [candidate.to_result_dict() for candidate in candidates]
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
                    signals = _scored_signal_dicts(reranked, graph)
                    score_type = (
                        "llm_pointwise_utility_0_100_within_discovered_shortlist"
                        if is_llm else "choice_probability_within_discovered_shortlist"
                    )
                else:
                    signals = [candidate.to_result_dict() for candidate in result.signals]
                    score_type = "no_rerank_singleton_or_empty_shortlist"
                graph_search = result.to_dict(graph)["search"]
            else:
                scored_signals = scorer.score(condition["condition"], candidates, graph)
                signals = _scored_signal_dicts(scored_signals, graph)
                score_type = (
                    "llm_pointwise_utility_0_100" if is_llm else
                    "choice_probability" if profile.runner == "strands-decider" else
                    "random_score" if profile.runner == "random" else "cross_encoder_score"
                )
                graph_search = None
            records.append({
                "result_format": "signal-discovery-v4",
                "condition_id": condition["id"], "condition": condition["condition"],
                "method": "catalog-rank" if args.list_only else args.method,
                "model": "schema-only" if args.list_only else profile.name,
                "model_id": model_name,
                "signal_catalog": args.catalog_view,
                "hop_limit": args.hop_limit,
                "score_type": score_type,
                "signals": signals,
                **({"oracle_root_table": oracle_roots[condition["id"]]} if is_oracle_root else {}),
                **({"random_seed": args.seed} if profile.runner == "random" else {}),
                **({"llm_config": _llm_provenance(scorer.config)} if is_llm else {}),
                **({"graph_search": graph_search, "graph_search_config": {
                    "max_decisions": args.max_decisions, "branch_width": args.branch_width,
                    "max_signals": args.max_signals,
                    "include_relationships": args.catalog_view == "path-aware" and not args.no_relationships,
                    "root_mode": "target_entity_proxy" if is_oracle_root else "model_selected",
                    "final_ranking": "LLM pointwise utility over discovered shortlist" if is_llm else "Decider over discovered shortlist",
                    "final_prompt_context": "explored schema view; no full candidate catalog",
                }} if graph_search is not None else {}),
            })
            # Persist completed conditions immediately so a long remote run can
            # resume without repeating already generated LLM calls.
            if args.resume and args.output is not None:
                _write_jsonl(records, args.output)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")

    _write_jsonl(records, args.output)
    if args.output is not None:
        run_label = "schema-only" if args.list_only else f"{args.method}/{profile.name}"
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


def _build_scorer(args: argparse.Namespace, profile: Any) -> Any:
    if profile.runner == "random":
        return RandomSignalScorer(seed=args.seed)
    if profile.runner == "minilm":
        assert profile.model_id is not None
        return MiniLMCrossEncoderScorer(
            model_name=profile.model_id,
            device=args.device,
            batch_size=args.batch_size,
        )
    if profile.runner == "strands-decider":
        assert profile.model_id is not None
        return DeciderSignalScorer(
            model_name=profile.model_id,
            backend="strands",
            device=args.device,
        )
    raise ValueError(f"unsupported model runner: {profile.runner}")


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


def _build_graph_navigator(
    args: argparse.Namespace,
    scorer: Any,
    model_name: str | None,
    shared_decider: Any | None,
    profile: Any,
    *,
    is_llm: bool,
) -> Any:
    common = {
        "max_hops": args.hop_limit,
        "max_decisions": args.max_decisions,
        "branch_width": args.branch_width,
        "max_signals": args.max_signals,
        "include_relationships": args.catalog_view == "path-aware" and not args.no_relationships,
    }
    if is_llm:
        assert isinstance(scorer, LLMPointwiseSignalScorer)
        return LLMGraphNavigator(scorer, **common)
    return DeciderGraphNavigator(
        model_name=model_name,
        backend=_decider_backend(profile.runner),
        device=args.device,
        client=shared_decider,
        **common,
    )


def _llm_provenance(config: LLMDiscoveryConfig) -> dict[str, object]:
    return {
        "model_family": config.model_family,
        "model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "seed": config.seed,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort,
        "max_attempts": config.max_attempts,
        "evaluation_note": "Discovery does not use human or LLM relevance labels.",
    }


def _scored_signal_dicts(scored_signals: Iterable[Any], graph: SchemaGraph) -> list[dict[str, object]]:
    """Serialize ordered scorer output with an explicit, stable one-based rank."""
    records = []
    for rank, scored in enumerate(scored_signals, start=1):
        record = dict(scored.to_result_dict())
        record["rank"] = rank
        records.append(record)
    return records


def _configure_model_cache(root: Path) -> None:
    """Set Hugging Face caches before vLLM imports a model.

    The caller supplies this path so the repository remains portable; on this
    cluster it can point to the shared /storage model cache rather than /home.
    """
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "transformers")


def _load_existing_records(output: Path) -> dict[str, dict[str, object]]:
    if not output.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(output.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{output}:{line_number}: invalid existing JSON: {error.msg}") from error
        if not isinstance(record, dict) or not isinstance(record.get("condition_id"), str):
            raise ValueError(f"{output}:{line_number}: expected a discovery record with condition_id")
        records[record["condition_id"]] = record
    return records


def _decider_backend(runner: str) -> str:
    if runner == "strands-decider":
        return "strands"
    raise ValueError(f"{runner!r} is not a typed-choice Decider runner")


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
