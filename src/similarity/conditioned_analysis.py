"""Load and compare conditioned rankings on matched held-out references."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts.compare_entity_rankings import compare_scores

METHODS = ("SimBench original", "SimBench bounded robust-z", "External baseline")


def _score(value, description):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid score for {description}")
    return float(value)


def _load_rankings(path, condition_id, table, condition, references, *, check_hash=None):
    scores = {}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if (str(row.get("condition_id")) != condition_id or row.get("entity_table") != table
                    or row.get("condition") != condition):
                raise ValueError(f"{path}: condition or entity table does not match the experiment")
            query = str(row["query_entity_id"])
            if query not in references:
                continue
            if query in scores:
                raise ValueError(f"{path}: duplicate reference {query}")
            candidates = {}
            for pair in row["pairs"]:
                candidate = str(pair["candidate_entity_id"])
                if candidate in candidates:
                    raise ValueError(f"{path}: duplicate candidate {candidate} for {query}")
                candidates[candidate] = _score(pair.get("similarity_score"), candidate)
            scores[query] = candidates
    if check_hash and digest.hexdigest() != check_hash:
        raise ValueError(f"{path}: input changed since the robust-z profile was fitted")
    if set(scores) != references:
        raise ValueError(f"{path}: missing held-out references: {sorted(references - set(scores))}")
    return scores


def load_condition(run_dir: Path, robust_dir: Path, condition_id: str, manifest: dict, *,
                   baseline_filename: str = "baseline.jsonl", baseline_label: str = METHODS[2]):
    """Load three rankings and coverage for every held-out query in one condition.

    The profile must match the experiment and the original SimBench output;
    all methods must describe the same complete candidate pool for each query.
    """
    if not baseline_label or baseline_label in METHODS[:2]:
        raise ValueError("baseline label must be nonempty and distinct from SimBench methods")
    methods = (*METHODS[:2], baseline_label)
    run_dir, robust_dir = Path(run_dir), Path(robust_dir)
    profile = json.loads((robust_dir / condition_id / "profile.json").read_text(encoding="utf-8"))
    condition = manifest["conditions"][condition_id]
    table = manifest["condition_tables"][condition_id]
    if (profile.get("format") != "simbench-experimental-scoring-v6"
            or profile.get("normalization") != "bounded-robust-z"
            or not profile.get("input_sha256")):
        raise ValueError(f"{condition_id}: invalid bounded robust-z profile or missing input fingerprint")
    if (profile.get("condition_id") != condition_id or profile.get("condition") != condition
            or profile.get("entity_table") != table):
        raise ValueError(f"{condition_id}: bounded robust-z profile does not match the condition or table")
    fingerprint = manifest.get("config", {}).get("dataset_fingerprint")
    if fingerprint and profile.get("dataset_fingerprint") != fingerprint:
        raise ValueError(f"{condition_id}: bounded robust-z profile belongs to a different dataset")
    heldout = {str(query) for query in profile.get("evaluation_queries", [])}
    train = {str(query) for query in profile.get("training_queries", [])}
    if not heldout or heldout & train:
        raise ValueError(f"{condition_id}: invalid or overlapping training and held-out references")

    robust, coverage = {}, {}
    with (robust_dir / condition_id / "pairs.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query, candidate = str(row["query"]), str(row["candidate"])
            if query not in heldout:
                raise ValueError(f"{condition_id}: bounded robust-z results include a non-held-out reference")
            candidates = robust.setdefault(query, {})
            if candidate in candidates:
                raise ValueError(f"{condition_id}: duplicate bounded robust-z pair {query}/{candidate}")
            score = _score(row.get("experimental"), candidate)
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError(f"{condition_id}: bounded robust-z score outside [0, 1] for {query}/{candidate}")
            candidates[candidate] = score
            value = _score(row.get("coverage"), f"coverage {query}/{candidate}")
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{condition_id}: invalid evidence coverage for {query}/{candidate}")
            coverage.setdefault(query, {})[candidate] = value
    if set(robust) != heldout:
        raise ValueError(f"{condition_id}: bounded robust-z results are missing held-out references")

    original = _load_rankings(run_dir / condition_id / "simbench.jsonl", condition_id, table,
                              condition, heldout, check_hash=profile.get("input_sha256"))
    baseline = _load_rankings(run_dir / condition_id / baseline_filename, condition_id, table,
                              condition, heldout)
    scores = {}
    for query in sorted(heldout):
        if set(original[query]) != set(baseline[query]) or set(original[query]) != set(robust[query]):
            raise ValueError(f"{condition_id}: candidate pools differ for reference {query}")
        scores[query] = {methods[0]: original[query], methods[1]: robust[query], methods[2]: baseline[query]}
    return {"profile": profile, "scores": scores, "coverage": coverage, "entity_table": table, "methods": methods}


def shared_scores(method_scores):
    """Restrict all methods to candidates scored by every method."""
    methods = tuple(method_scores)
    if not methods:
        return {}
    first = method_scores[methods[0]]
    common = sorted(candidate for candidate in first
                    if all(method_scores[method].get(candidate) is not None for method in methods))
    return {method: {candidate: method_scores[method][candidate] for candidate in common}
            for method in methods}


def compare_condition(condition_id: str, loaded: dict, top_k: int):
    """Compute all pairwise agreements on the same three-way candidate set."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    rows = []
    methods = loaded["methods"]
    for query, method_scores in loaded["scores"].items():
        common = shared_scores(method_scores)
        total = len(method_scores[METHODS[0]])
        for method_a, method_b in itertools.combinations(methods, 2):
            metrics = compare_scores(common[method_a], common[method_b], top_k)
            metrics["candidates"] = total
            rows.append({"condition_id": condition_id, "entity_table": loaded["entity_table"],
                         "query_entity_id": query, "method_a": method_a, "method_b": method_b,
                         **metrics})
    return rows
