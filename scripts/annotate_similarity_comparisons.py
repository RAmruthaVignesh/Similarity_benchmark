#!/usr/bin/env python3
"""Blind human comparison of original and experimental top candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.datasets.local_relbench_dataset import LocalRelBenchDataset


def stable_id(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def unique_top(rows, column):
    scored = [row for row in rows if row.get(column) is not None]
    if not scored:
        return None
    best = max(row[column] for row in scored)
    winners = [str(row["candidate"]) for row in scored if row[column] == best]
    return winners[0] if len(winners) == 1 else None


def cases_for_condition(root, cid, seed):
    profile_path = root / cid / "profile.json"
    pairs_path = root / cid / "pairs.jsonl"
    profile = json.loads(profile_path.read_text())
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    groups = defaultdict(list)
    with pairs_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                groups[str(row["query"])].append(row)
    cases = []
    for query, rows in groups.items():
        original, experimental = unique_top(rows, "baseline"), unique_top(rows, "experimental")
        if original is None or experimental is None:
            continue
        control = None
        if original == experimental:
            # Agreement control: compare the shared top result with the
            # original ranking's unique next result, without attributing that
            # next result to the experimental method.
            control = unique_top([row for row in rows if str(row["candidate"]) != original], "baseline")
            if control is None:
                continue
            first, second, kind = original, control, "agreement_control"
        else:
            first, second, kind = original, experimental, "disagreement"
        case_id = stable_id(profile_hash, cid, query, first, second, kind)
        swap = int(stable_id(seed, case_id)[:8], 16) % 2
        a, b = (second, first) if swap else (first, second)
        cases.append({"case_id": case_id, "condition_id": cid, "condition": profile["condition"],
                      "entity_table": profile["entity_table"], "reference": query,
                      "candidate_a": a, "candidate_b": b, "case_kind": kind,
                      "original_candidate": original, "experimental_candidate": experimental,
                      "control_candidate": control, "profile_sha256": profile_hash})
    random.Random(f"{seed}:{cid}").shuffle(cases)
    return cases


def format_value(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)


def entity_details(dataset, tables, table, entity_id):
    if table not in tables:
        frame = dataset.get_table(table).to_pandas()
        primary = dataset.get_primary_key(table)
        frame.index = frame[primary].map(str) if primary else frame.index.map(str)
        tables[table] = frame
    frame = tables[table]
    if str(entity_id) not in frame.index:
        raise ValueError(f"Entity {entity_id} missing from {table}")
    row = frame.loc[str(entity_id)]
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Duplicate entity ID {entity_id} in {table}")
    return "\n".join(f"  {name}: {format_value(value)}" for name, value in row.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/rel-amazon/similarity/experimental-scoring-bounded-robust-z")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/dev/rel-amazon")
    parser.add_argument("--output", type=Path, default=ROOT / "annotations/rel-amazon/entity-comparisons/bounded-robust-z-pairwise.jsonl")
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=30, help="maximum cases shown in this session")
    args = parser.parse_args()
    if args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    try:
        known = {path.name for path in args.results_dir.glob("condition-*")
                 if (path / "profile.json").is_file()}
        chosen = set(args.condition_id or known)
        if chosen - known:
            raise ValueError(f"Unknown conditions: {sorted(chosen - known)}")
        completed = set()
        if args.output.is_file():
            with args.output.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        completed.add(json.loads(line)["case_id"])
        by_condition = {
            cid: [case for case in cases_for_condition(args.results_dir, cid, args.seed)
                  if case["case_id"] not in completed]
            for cid in sorted(chosen)
            if (args.results_dir / cid / "profile.json").is_file()
        }
        # Round-robin conditions to avoid spending the session on one condition.
        cases = []
        while any(by_condition.values()) and len(cases) < args.max_pairs:
            for key in sorted(by_condition):
                if by_condition[key] and len(cases) < args.max_pairs:
                    cases.append(by_condition[key].pop())
        print(f"{len(cases)} unannotated cases selected; A/B method identity is hidden.")
        if not cases:
            return 0
        dataset = LocalRelBenchDataset(args.dataset_root)
        tables = {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            for number, case in enumerate(cases, 1):
                print("\n" + "=" * 80)
                print(f"Pair {number}/{len(cases)} | {case['condition_id']}")
                print(f"CONDITION: {case['condition']}\n")
                for title, entity_id in (("REFERENCE", case["reference"]),
                                         ("CANDIDATE A", case["candidate_a"]),
                                         ("CANDIDATE B", case["candidate_b"])):
                    print(f"{title} — {case['entity_table']} {entity_id}")
                    print(entity_details(dataset, tables, case["entity_table"], entity_id))
                    print()
                while True:
                    answer = input("Which fits the condition better? [a/b/tie/s/q]: ").strip().lower()
                    if answer in {"a", "b", "tie", "s", "q"}:
                        break
                if answer == "q":
                    break
                if answer == "s":
                    continue
                rationale = input("Rationale (optional): ").strip()
                winner = None if answer == "tie" else case["candidate_" + answer]
                output = dict(case, preference=answer, preferred_candidate=winner, rationale=rationale,
                              annotation_type="blind_pairwise", seed=args.seed)
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
                stream.flush()
                print("Saved.")
        print(f"Annotations: {args.output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
