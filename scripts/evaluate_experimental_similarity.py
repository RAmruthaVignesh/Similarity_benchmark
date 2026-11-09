#!/usr/bin/env python3
"""Fit and evaluate experimental scores on disjoint reference-query splits."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.similarity.score_diagnostics import split_queries, ranking_comparison, tie_top_membership
from src.similarity.experimental_scoring import (
    load_components, fit_profile, score_components,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/rel-amazon/similarity/conditioned-comparison")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/rel-amazon/similarity/experimental-scoring-bounded-robust-z")
    parser.add_argument("--condition-id", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--missing-policy", choices=("reference", "drop"), default="reference")
    parser.add_argument("--coverage-penalty", type=float, default=0.0)
    args = parser.parse_args()
    try:
        manifest_path = args.input_dir / "experiment.json"
        manifest = json.loads(manifest_path.read_text())
        available = [cid for cid in sorted(manifest["conditions"]) if (args.input_dir / cid / "simbench.jsonl").is_file()]
        chosen = args.condition_id or available
        if set(chosen) - set(available):
            raise ValueError(f"Unknown or unavailable conditions: {sorted(set(chosen) - set(available))}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for cid in chosen:
            print(f"Processing {cid} (bounded robust-z)", flush=True)
            out = args.output_dir / cid
            if (out / "profile.json").exists():
                raise ValueError(f"Results already exist for {cid}; choose a new --output-dir")
            source = args.input_dir / cid / "simbench.jsonl"
            frame, metadata = load_components(source)
            train_ids, test_ids = split_queries(frame, args.seed)
            train = frame[frame["query"].isin(train_ids)]
            test = frame[frame["query"].isin(test_ids)]
            profile = fit_profile(train, metadata, coverage_penalty=args.coverage_penalty,
                                  missing_policy=args.missing_policy)
            profile["dataset_fingerprint"] = manifest["config"]["dataset_fingerprint"]
            profile["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            profile["split_seed"] = args.seed
            profile["evaluation_queries"] = sorted(test_ids)
            scores, _ = score_components(test, profile)
            valid = test[test["raw_score"].notna()].copy()
            valid["effective_weight"] = valid["base_weight"] * valid["confidence"]
            valid["weighted"] = valid["raw_score"] * valid["effective_weight"]
            aggregate = valid.groupby(["query", "candidate"]).agg(n=("weighted", "sum"), d=("effective_weight", "sum"))
            baseline = (aggregate["n"] / aggregate["d"].where(aggregate["d"] > 0)).rename("score")
            metrics = ranking_comparison(baseline, scores["score"], args.top_k)
            pair_data = test[["query", "candidate"]].drop_duplicates().set_index(["query", "candidate"])
            pair_data["baseline"] = baseline
            pair_data["experimental"] = scores["score"]
            pair_data["coverage"] = scores["coverage"]
            pair_data.reset_index(inplace=True)
            top_rows = []
            for query, group in pair_data.groupby("query", sort=True):
                for mode in ("baseline", "experimental"):
                    values = group.set_index("candidate")[mode]
                    memberships = tie_top_membership(values, args.top_k)
                    chosen_rows = group.set_index("candidate").reindex(memberships.index)
                    mean_coverage = float((memberships * chosen_rows["coverage"]).sum() / memberships.sum()) if memberships.sum() else np.nan
                    top_rows.append({"query": query, "mode": mode, "mean_coverage": mean_coverage})
            top = pd.DataFrame(top_rows)
            out.mkdir(parents=True, exist_ok=True)
            (out / "profile.json").write_text(json.dumps(profile, ensure_ascii=False) + "\n")
            pair_data.to_json(out / "pairs.jsonl", orient="records", lines=True)
            metrics.to_json(out / "ranking-comparison.jsonl", orient="records", lines=True)
            top.to_json(out / "top-coverage.jsonl", orient="records", lines=True)
            summary = {
                "method": "bounded-robust-z", "condition_id": cid, "condition": metadata["condition"],
                "entity_table": metadata["entity_table"], "training_references": len(train_ids),
                "held_out_references": len(test_ids), "pairs": len(pair_data),
                "mean_spearman": metrics["spearman"].mean(),
                "mean_top_k_jaccard": metrics["top_jaccard"].mean(),
                "mean_comparable_fraction": metrics["coverage"].mean(),
                "baseline_top_coverage": top.loc[top["mode"].eq("baseline"), "mean_coverage"].mean(),
                "experimental_top_coverage": top.loc[top["mode"].eq("experimental"), "mean_coverage"].mean(),
                "normalization": "bounded-robust-z", "missing_policy": args.missing_policy,
                "coverage_penalty": args.coverage_penalty,
            }
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        summaries.sort(key=lambda row: row["condition_id"])
        (args.output_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n")
        print(f"Wrote {len(summaries)} bounded robust-z condition evaluations to {args.output_dir}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
