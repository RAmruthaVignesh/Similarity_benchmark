#!/usr/bin/env python3
"""Turn selected schema signals into transparent entity-pair similarity scores.

This is the initial signal-specific implementation of proposal section 3.2.
It deliberately stops before learned weighting or score-to-grade conversion:
the output preserves every selected signal's type-appropriate component score
so those later decisions can be validated separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import (  # noqa: E402
    SignalCandidate,
    candidate_from_dict,
    enumerate_signal_candidates,
)
from src.similarity.signal_measures import MEASURE_VARIANTS, SignalMeasureExecutor  # noqa: E402
from src.similarity.cohort import load_entity_ids  # noqa: E402
from src.similarity.discovery_selection import select_discovery_signals  # noqa: E402

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"
DEFAULT_CONDITIONS = REPO_ROOT / "conditions" / "rel-amazon" / "conditions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument(
        "--entity-table",
        required=True,
        help="table containing the query and candidate entities to compare",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--signal-id",
        action="append",
        help="selected candidate ID; repeat for several signals",
    )
    source.add_argument(
        "--discovery-results",
        type=Path,
        help="signal-discovery JSONL whose top-ranked signals should be used",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--top-signals",
        type=int,
        default=None,
        help=(
            "how many highest-ranked signals compatible with --entity-table to take "
            "from --discovery-results (default: 3)"
        ),
    )
    selection.add_argument(
        "--min-utility-score", type=int, choices=range(11),
        help="select every compatible signal with 0–10 discovery utility at or above this score",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        required=True,
        help="primary-key value, or frozen row position for a table without a primary key; repeat for several queries",
    )
    parser.add_argument(
        "--candidate-ids-file", type=Path,
        help="JSON array of candidate entity IDs; omit to rank the full table",
    )
    parser.add_argument(
        "--signal-weight",
        action="append",
        default=[],
        metavar="SIGNAL_ID=WEIGHT",
        help="non-negative composition weight; omitted signals receive weight 1",
    )
    parser.add_argument(
        "--measure-variant",
        choices=MEASURE_VARIANTS,
        default="auto",
        help="apply one explicit measure variant to all selected signals (default: auto)",
    )
    parser.add_argument(
        "--signal-measure",
        action="append",
        default=[],
        metavar="SIGNAL_ID=VARIANT",
        help="override the global variant for one signal; repeat for multiple signals",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="write only this many highest-scoring candidates per query (default: all)",
    )
    parser.add_argument("--hop-limit", type=int, default=2)
    parser.add_argument(
        "--model-cache-root",
        type=Path,
        help="Hugging Face hub cache containing the optional local text embedding model",
    )
    parser.add_argument(
        "--text-embedding-model",
        help="optional local Hugging Face text-embedding model ID (defaults to E5-small-v2 for e5-vllm)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="number of text values embedded per vLLM batch (default: 64)",
    )
    parser.add_argument("--experimental-profile", type=Path, help="apply a previously fitted experimental scoring profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.hop_limit < 0:
        parser.error("--hop-limit must be non-negative")
    if args.top_signals is not None and args.top_signals <= 0:
        parser.error("--top-signals must be positive")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")

    try:
        condition = _load_condition(args.conditions, args.condition_id)
        dataset = LocalRelBenchDataset(args.dataset_root)
        graph = SchemaGraph.from_manifest(dataset.schema)
        selected = _selected_signals(args, dataset, graph)
        if any(signal.source_table != args.entity_table for signal in selected):
            wrong = [signal.id for signal in selected if signal.source_table != args.entity_table]
            raise ValueError(
                "all selected signals must start from --entity-table "
                f"{args.entity_table!r}; incompatible IDs: {', '.join(wrong)}"
            )
        weights = _parse_weights(args.signal_weight, selected)
        measure_variants = _parse_measure_variants(args.signal_measure, selected)
        executor = SignalMeasureExecutor(
            dataset,
            graph,
            text_embedding_model=args.text_embedding_model or "intfloat/e5-small-v2",
            model_cache_root=args.model_cache_root,
            embedding_batch_size=args.embedding_batch_size,
        )
        materialized = []
        for signal in selected:
            print(f"Materializing signal {len(materialized) + 1}/{len(selected)}: {signal.id}", flush=True)
            materialized.append(executor.materialize(
                signal,
                entity_table=args.entity_table,
                measure_variant=measure_variants.get(signal.id, args.measure_variant),
            ))
        candidate_ids = load_entity_ids(args.candidate_ids_file) if args.candidate_ids_file else None
        profile = None
        if args.experimental_profile is not None:
            if args.signal_weight:
                raise ValueError("experimental profiles already contain weights; fit a new profile to change them")
            profile = json.loads(args.experimental_profile.read_text())
            from src.similarity.entity_embeddings import dataset_fingerprint
            if profile.get("dataset_fingerprint") != dataset_fingerprint(dataset):
                raise ValueError("experimental profile dataset fingerprint does not match the current dataset")
            if (profile.get("condition_id") != condition["id"] or
                    profile.get("condition") != condition["condition"] or
                    profile.get("entity_table") != args.entity_table):
                raise ValueError("experimental profile condition or entity table does not match")
            from src.similarity.experimental_scoring import rescore_pairs
        records = []
        for query_index, requested_query_id in enumerate(args.query_id, 1):
            scored = executor.score_query(requested_query_id, materialized, weights=weights, candidate_entity_ids=candidate_ids)
            if profile is not None:
                pair_records = rescore_pairs(scored, profile, materialized)
            else:
                pair_records = [pair.to_dict() for pair in scored]
            if args.top_k is not None:
                pair_records = pair_records[:args.top_k]
            query_id = pair_records[0]["query_entity_id"] if pair_records else requested_query_id
            records.append(
                {
                    "result_format": "signal-similarity-v1",
                    "condition_id": condition["id"],
                    "condition": condition["condition"],
                    "entity_table": args.entity_table,
                    "entity_id_kind": materialized[0].entity_id_kind,
                    "query_entity_id": query_id,
                    "signals": [
                        {
                            "candidate": signal.candidate.to_result_dict(),
                            "value_kind": signal.value_kind,
                            "measure": signal.measure,
                            "weight": weights.get(signal.candidate.id, 1.0),
                        }
                        for signal in materialized
                    ],
                    "pairs": pair_records,
                    **({"scoring_mode": "experimental", "scoring_profile": str(args.experimental_profile.resolve())} if profile is not None else {}),
                }
            )
            if query_index == 1 or query_index % 10 == 0 or query_index == len(args.query_id):
                print(f"Ranked {query_index}/{len(args.query_id)} SimBench references", flush=True)
        _write_jsonl(records, args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        f"wrote {len(records)} query ranking(s) with {len(selected)} selected signal(s) to {args.output}"
    )
    return 0


def _selected_signals(
    args: argparse.Namespace, dataset: LocalRelBenchDataset, graph: SchemaGraph
) -> list[SignalCandidate]:
    if args.discovery_results is not None:
        record = next(
            (
                row
                for row in _load_jsonl(args.discovery_results)
                if row.get("condition_id") == args.condition_id
            ),
            None,
        )
        if record is None:
            raise KeyError(
                f"condition {args.condition_id!r} was not found in {args.discovery_results}"
            )
        raw_signals = record.get("signals")
        if not isinstance(raw_signals, list):
            raise ValueError("discovery record has no signal list")
        selected = select_discovery_signals(
            record, args.entity_table,
            top_signals=(args.top_signals or 3) if args.min_utility_score is None else None,
            min_utility_score=args.min_utility_score,
        )
        return [candidate_from_dict(item) for item in selected]

    # Signal IDs are only valid for the selected entity root during scoring.
    # Enumerating every root here needlessly rebuilds the full catalog just to
    # validate a handful of selected IDs.
    catalog = enumerate_signal_candidates(
        dataset, graph, source_tables=[args.entity_table], max_hops=args.hop_limit
    )
    by_id = {candidate.id: candidate for candidate in catalog}
    missing = [signal_id for signal_id in args.signal_id if signal_id not in by_id]
    if missing:
        raise KeyError(f"unknown signal ID(s): {', '.join(missing)}")
    return [by_id[signal_id] for signal_id in args.signal_id]


def _parse_weights(raw_weights: Iterable[str], signals: Iterable[SignalCandidate]) -> dict[str, float]:
    known = {signal.id for signal in signals}
    weights = {}
    for raw in raw_weights:
        signal_id, separator, raw_weight = raw.rpartition("=")
        if not separator or not signal_id:
            raise ValueError(f"invalid --signal-weight {raw!r}; expected SIGNAL_ID=WEIGHT")
        if signal_id not in known:
            raise ValueError(f"weight supplied for unselected signal {signal_id!r}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError("signal weights must be non-negative")
        weights[signal_id] = weight
    return weights


def _parse_measure_variants(
    raw_variants: Iterable[str], signals: Iterable[SignalCandidate]
) -> dict[str, str]:
    known = {signal.id for signal in signals}
    variants: dict[str, str] = {}
    for raw in raw_variants:
        signal_id, separator, variant = raw.rpartition("=")
        if not separator or not signal_id:
            raise ValueError(f"invalid --signal-measure {raw!r}; expected SIGNAL_ID=VARIANT")
        if signal_id not in known:
            raise ValueError(f"measure variant supplied for unselected signal {signal_id!r}")
        if variant not in MEASURE_VARIANTS:
            raise ValueError(
                f"unknown measure variant {variant!r}; choose from {', '.join(MEASURE_VARIANTS)}"
            )
        if signal_id in variants:
            raise ValueError(f"measure variant specified more than once for {signal_id!r}")
        variants[signal_id] = variant
    return variants


def _load_condition(path: Path, condition_id: str) -> dict[str, str]:
    for record in _load_jsonl(path):
        if record.get("id") == condition_id:
            condition = record.get("condition")
            if not isinstance(condition, str):
                raise ValueError(f"condition {condition_id!r} has no text")
            return {"id": condition_id, "condition": condition}
    raise KeyError(f"condition {condition_id!r} was not found in {path}")


def _load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield record


def _write_jsonl(records: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str, sort_keys=True))
            handle.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
