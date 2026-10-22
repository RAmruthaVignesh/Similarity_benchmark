#!/usr/bin/env python3
"""Collect blind, resumable human relevance judgments for schema signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402

RUBRIC_VERSION = "signal-relevance-0.1"
LABELS = {
    0: "Not useful for this intent.",
    1: "Weakly or indirectly useful; likely needs other signals.",
    2: "Useful evidence for filtering or ranking results for this intent.",
    3: "Highly useful; directly captures a central part of the intent.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser("annotate", help="annotate intent/signal pairs")
    annotate.add_argument("--conditions", type=Path, required=True, help="conditions JSONL")
    annotate.add_argument("--dataset-root", type=Path, required=True, help="local dataset with manifest and parquet tables")
    annotate.add_argument("--annotator", required=True, help="stable annotator ID (use a pseudonym if preferred)")
    annotate.add_argument("--output", type=Path, required=True, help="this annotator's JSONL file; existing entries resume")
    annotate.add_argument("--hop-limit", type=int, default=2)
    annotate.add_argument("--source-table", action="append", dest="source_tables")
    annotate.add_argument("--seed", type=int, default=0, help="base seed for reproducible per-annotator shuffling")
    annotate.add_argument("--include-structural-columns", action="store_true")

    compare = subparsers.add_parser("compare", help="compare annotator coverage and ordinal agreement")
    compare.add_argument("files", nargs="+", type=Path, help="two or more annotator JSONL files")
    compare.add_argument("--disagreements", type=Path, help="optional JSONL export of pairs with different grades")

    args = parser.parse_args()
    if args.command == "annotate":
        if args.hop_limit < 0:
            parser.error("--hop-limit must be non-negative")
        run_annotation(args)
    else:
        if len(args.files) < 2:
            parser.error("compare requires at least two annotator files")
        compare_annotations(args)
    return 0


def load_conditions(path: Path) -> list[dict[str, str]]:
    conditions = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("condition"), str):
                raise ValueError(f"{path}:{line_number}: expected string 'id' and 'condition' fields")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate condition ID {row['id']!r}")
            seen.add(row["id"])
            conditions.append({
                "id": row["id"],
                "condition": row["condition"],
            })
    if not conditions:
        raise ValueError(f"no conditions found in {path}")
    return conditions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    return rows


def run_annotation(args: argparse.Namespace) -> None:
    conditions = load_conditions(args.conditions)
    print(f"Loaded {len(conditions)} conditions. Loading dataset manifest...", flush=True)
    dataset = LocalRelBenchDataset(args.dataset_root)
    graph = SchemaGraph.from_manifest(dataset.schema)
    print(
        f"Enumerating schema signals for {len(dataset.get_tables())} tables "
        f"(maximum {args.hop_limit} relationship hops)...",
        flush=True,
    )
    candidates = enumerate_signal_candidates(
        dataset,
        graph,
        source_tables=args.source_tables,
        max_hops=args.hop_limit,
        include_structural_columns=args.include_structural_columns,
    )
    if not candidates:
        raise ValueError("schema candidate enumeration returned no signals")

    prior_rows = read_jsonl(args.output)
    for row in prior_rows:
        if row.get("annotator") != args.annotator:
            raise ValueError(f"{args.output} contains a different annotator ID; use a separate file")
    by_pair = {(row.get("condition_id"), row.get("candidate_id")): row for row in prior_rows}
    shuffle_key = hashlib.sha256(args.annotator.encode("utf-8")).digest()
    annotator_seed = int.from_bytes(shuffle_key[:8], "big") ^ args.seed
    rng = random.Random(annotator_seed)
    condition_order = list(conditions)
    rng.shuffle(condition_order)
    tasks = []
    for condition in condition_order:
        condition_tasks = [
            (condition, candidate)
            for candidate in candidates
            if by_pair.get((condition["id"], candidate.id), {}).get("status") != "complete"
        ]
        rng.shuffle(condition_tasks)
        tasks.extend(condition_tasks)
    print(f"Annotator: {args.annotator}")
    print(f"Conditions: {len(conditions)} | Signals per condition: {len(candidates)} | Total pairs: {len(conditions) * len(candidates)}")
    print(f"Already complete: {len(conditions) * len(candidates) - len(tasks)} | Remaining: {len(tasks)}")
    print("Rate usefulness for this intent, not whether the signal is required. 'u' = unsure, 's' = skip for now, 'q' = quit.")

    for index, (condition, candidate) in enumerate(tasks, 1):
        print("\n" + "=" * 76)
        print(f"Pair {index}/{len(tasks)} | {condition['id']}")
        print(f"INTENT: {condition['condition']}")
        print(f"\nSIGNAL: {candidate.explain_for_human(graph)}")
        print("\nPATH:")
        print(candidate.human_diagram(graph))
        print(f"\nTECHNICAL DEFINITION: {candidate.describe(graph)}")
        print("\nGrades:")
        for grade, description in LABELS.items():
            print(f"  {grade}: {description}")
        while True:
            answer = input("Grade [0/1/2/3, u, s, q]: ").strip().lower()
            if answer in {"0", "1", "2", "3"}:
                grade = int(answer)
                rationale = input("Optional rationale (Enter to skip): ").strip()
                row = {
                    "rubric_version": RUBRIC_VERSION,
                    "annotator": args.annotator,
                    "condition_id": condition["id"],
                    "condition": condition["condition"],
                    "candidate_id": candidate.id,
                    "candidate": candidate.to_dict(graph),
                    "grade": grade,
                    "status": "complete",
                    "rationale": rationale,
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                    "candidate_config": {
                        "hop_limit": args.hop_limit,
                        "source_tables": list(args.source_tables or dataset.get_tables()),
                        "include_structural_columns": args.include_structural_columns,
                    },
                }
                by_pair[(condition["id"], candidate.id)] = row
                write_jsonl_atomic(args.output, by_pair.values())
                break
            if answer == "u":
                by_pair[(condition["id"], candidate.id)] = {
                    "rubric_version": RUBRIC_VERSION,
                    "annotator": args.annotator,
                    "condition_id": condition["id"],
                    "condition": condition["condition"],
                    "candidate_id": candidate.id,
                    "candidate": candidate.to_dict(graph),
                    "grade": None,
                    "status": "unsure",
                    "rationale": input("Why unsure? (optional): ").strip(),
                    "annotated_at": datetime.now(timezone.utc).isoformat(),
                }
                write_jsonl_atomic(args.output, by_pair.values())
                break
            if answer == "s":
                break
            if answer == "q":
                print(f"Saved progress to {args.output}")
                return
            print("Enter 0, 1, 2, 3, u, s, or q.")
    print(f"Annotation session complete. Saved {len(by_pair)} records to {args.output}")


def write_jsonl_atomic(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: (item["condition_id"], item["candidate_id"])):
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def compare_annotations(args: argparse.Namespace) -> None:
    annotators: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for path in args.files:
        rows = read_jsonl(path)
        names = {row.get("annotator") for row in rows}
        if len(names) != 1 or None in names:
            raise ValueError(f"{path}: expected exactly one annotator ID")
        name = str(next(iter(names)))
        if name in annotators:
            raise ValueError(f"duplicate annotator ID: {name}")
        annotators[name] = {
            (row.get("condition_id"), row.get("candidate_id")): row
            for row in rows
            if row.get("status") == "complete" and row.get("grade") in LABELS
        }

    print("Completed judgments:")
    for name, judgments in annotators.items():
        print(f"  {name}: {len(judgments)}")
    disagreements = []
    names = sorted(annotators)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = annotators[left_name], annotators[right_name]
            common = sorted(left.keys() & right.keys())
            exact = sum(left[key]["grade"] == right[key]["grade"] for key in common)
            kappa = weighted_kappa(
                [left[key]["grade"] for key in common],
                [right[key]["grade"] for key in common],
            )
            print(
                f"{left_name} vs {right_name}: overlap={len(common)}, "
                f"exact agreement={exact / len(common):.3f} ({exact}/{len(common)}), "
                f"linear weighted kappa={kappa if kappa is not None else 'undefined'}"
                if common
                else f"{left_name} vs {right_name}: no overlapping completed judgments"
            )
            for key in common:
                if left[key]["grade"] != right[key]["grade"]:
                    disagreements.append({
                        "condition_id": key[0],
                        "candidate_id": key[1],
                        "candidate": left[key].get("candidate"),
                        "condition": left[key].get("condition"),
                        left_name: left[key]["grade"],
                        right_name: right[key]["grade"],
                    })
    print(f"Pairs with differing grades across at least one annotator pair: {len(disagreements)}")
    if args.disagreements:
        write_jsonl_atomic(args.disagreements, disagreements)
        print(f"Wrote disagreement list to {args.disagreements}")


def weighted_kappa(left: list[int], right: list[int]) -> float | None:
    """Linear-weighted Cohen kappa for ordinal grades 0..3."""
    if not left:
        return None
    categories = range(4)
    n = len(left)
    observed = sum(abs(a - b) / 3 for a, b in zip(left, right)) / n
    left_counts = [sum(value == category for value in left) for category in categories]
    right_counts = [sum(value == category for value in right) for category in categories]
    expected = sum(
        left_counts[a] * right_counts[b] * abs(a - b) / 3
        for a in categories for b in categories
    ) / (n * n)
    return 1.0 if expected == 0 and observed == 0 else (None if expected == 0 else 1 - observed / expected)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
