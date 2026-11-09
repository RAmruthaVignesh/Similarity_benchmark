"""Experimental, fitted aggregation of saved per-signal similarities.

Profiles are fitted on explicit reference queries; they never fit on the
queries being scored unless the caller deliberately supplies overlapping data.
This is distribution normalization, not calibration to human relevance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

def load_components(path: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    meta = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if meta is None:
                meta = {key: record[key] for key in ("condition_id", "condition", "entity_table", "entity_id_kind")}
                meta["signals"] = record["signals"]
            elif any(record[key] != meta[key] for key in ("condition_id", "condition", "entity_table")):
                raise ValueError("Input contains multiple conditions or entity tables")
            for pair in record["pairs"]:
                for component in pair["components"]:
                    rows.append({
                        "query": str(record["query_entity_id"]),
                        "candidate": str(pair["candidate_entity_id"]),
                        "signal": component["signal_id"],
                        "measure": component["measure"],
                        "raw_score": component["score"],
                        "confidence": component.get("confidence", 1.0),
                        "base_weight": component.get("weight", 1.0),
                    })
    if not rows or meta is None:
        raise ValueError("No saved signal components found")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["query", "candidate", "signal"]).any():
        raise ValueError("Duplicate signal component for query/candidate")
    return frame, meta


def _robust_center_scale(values: np.ndarray) -> tuple[float, float, str]:
    """Return a median center and robust scale, with defined degenerate fallbacks."""
    center = float(np.median(values))
    deviations = np.abs(values - center)
    mad = float(np.median(deviations))
    if mad > 0:
        return center, 1.4826 * mad, "mad"
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr = float(q75 - q25)
    if iqr > 0:
        return center, iqr / 1.349, "iqr"
    nonzero = deviations[deviations > 0]
    if len(nonzero):
        return center, float(np.median(nonzero)), "nonzero-median-deviation"
    return center, 1.0, "constant"


def _bounded_robust_z(values):
    """Map robust z-scores monotonically into [0, 1], with zero mapped to 0.5."""
    values = np.asarray(values, dtype=float)
    return 0.5 + values / (2.0 * (1.0 + np.abs(values)))


def fit_profile(frame: pd.DataFrame, meta: dict, *, coverage_penalty=0.0,
                missing_policy="reference"):
    if missing_policy not in {"reference", "drop"}:
        raise ValueError("missing_policy must be reference or drop")
    if not 0 <= coverage_penalty <= 1:
        raise ValueError("coverage_penalty must be in [0,1]")
    info = {row["candidate"]["id"]: row for row in meta["signals"]}
    profile = {
        "format": "simbench-experimental-scoring-v6",
        "condition_id": meta["condition_id"], "condition": meta["condition"],
        "entity_table": meta["entity_table"], "method": "bounded-robust-z",
        "coverage_penalty": coverage_penalty,
        "normalization": "bounded-robust-z", "missing_policy": missing_policy,
        "score_transform": "0.5 + z / (2 * (1 + abs(z)))",
        "score_range": [0.0, 1.0],
        "training_queries": sorted(frame["query"].unique().tolist()),
        "signals": {},
    }
    for signal, group in frame.groupby("signal", sort=True):
        measure = str(group["measure"].iloc[0])
        observed = pd.to_numeric(group["raw_score"], errors="coerce").dropna().to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            raise ValueError(f"No training evidence available for {signal}")
        center, scale, scale_method = _robust_center_scale(observed)
        robust_z = (observed - center) / scale
        transformed = _bounded_robust_z(robust_z)
        profile["signals"][signal] = {
            "measure": measure, "value_kind": info.get(signal, {}).get("value_kind"),
            "weight": 1.0, "normalization": "bounded-robust-z",
            "center": center, "scale": scale, "scale_method": scale_method,
            "missing_reference": float(np.median(transformed)),
            "training_observations": int(len(group)), "fit_values": int(len(observed)),
        }
    return profile


def score_components(frame: pd.DataFrame, profile: dict):
    """Return per-pair scores and a component frame with explicit evidence."""
    if profile.get("format") != "simbench-experimental-scoring-v6":
        raise ValueError("Invalid experimental scoring profile")
    if set(frame["signal"]) != set(profile["signals"]):
        raise ValueError("Profile and scored signals differ")
    out = frame.copy()
    out["experimental_score"] = np.nan
    out["experimental_weight"] = 0.0
    out["usable_confidence"] = 0.0
    out["imputed"] = False
    for signal, indices in out.groupby("signal", sort=False).groups.items():
        spec = profile["signals"][signal]
        subset = out.loc[indices]
        values = pd.to_numeric(subset["raw_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.notna()
        normalized = values.copy()
        if spec["scale"] <= 0:
            raise ValueError(f"Invalid fitted robust scale for {signal}")
        robust_z = (values[valid] - spec["center"]) / spec["scale"]
        normalized.loc[valid] = _bounded_robust_z(robust_z)
        confidence = pd.to_numeric(subset["confidence"], errors="coerce").fillna(0).clip(0, 1)
        confidence = confidence.where(valid, 0.0)
        neutral = spec["missing_reference"]
        if profile.get("missing_policy", "reference") == "drop":
            final_value = normalized.fillna(0.0)
            effective_weight = spec["weight"] * confidence
        else:
            final_value = confidence * normalized.fillna(neutral) + (1 - confidence) * neutral
            effective_weight = spec["weight"]
        out.loc[indices, "experimental_score"] = final_value
        out.loc[indices, "experimental_weight"] = effective_weight
        out.loc[indices, "nominal_weight"] = 1.0
        out.loc[indices, "usable_confidence"] = confidence
        out.loc[indices, "imputed"] = ~valid
    out["weighted_score"] = out["experimental_score"] * out["experimental_weight"]
    out["available_weight"] = out["usable_confidence"] * out["nominal_weight"]
    grouped = out.groupby(["query", "candidate"], sort=False).agg(
        weighted=("weighted_score", "sum"), total_weight=("experimental_weight", "sum"),
        nominal=("nominal_weight", "sum"), available=("available_weight", "sum"))
    coverage = (grouped["available"] / grouped["nominal"]).rename("coverage")
    base = grouped["weighted"] / grouped["total_weight"].where(grouped["total_weight"] > 0)
    final = (base - profile["coverage_penalty"] * (1.0 - coverage)).clip(0.0, 1.0)
    final = final.where(coverage > 0).rename("score")
    return pd.concat([final, coverage], axis=1), out


def rescore_pairs(pairs, profile, materialized):
    """Apply a fitted profile to freshly computed PairSimilarity objects."""
    by_signal = {signal.candidate.id: signal for signal in materialized}
    if set(by_signal) != set(profile["signals"]):
        raise ValueError("Profile signal IDs do not match the selected signals")
    for signal_id, item in by_signal.items():
        if item.measure != profile["signals"][signal_id]["measure"]:
            raise ValueError(f"Similarity measure changed for {signal_id}")
    frame_rows = []
    originals = []
    for pair in pairs:
        record = pair.to_dict()
        originals.append(record)
        candidate = pair.candidate_entity_id
        for component in record["components"]:
            signal_id = component["signal_id"]
            frame_rows.append({
                "query": str(pair.query_entity_id), "candidate": str(candidate), "signal": signal_id,
                "measure": component["measure"], "raw_score": component.get("score"),
                "confidence": component.get("confidence", 1.0), "base_weight": 1.0,
            })
    if not frame_rows:
        return []
    frame = pd.DataFrame(frame_rows)
    totals, scored = score_components(frame, profile)
    lookups = {(str(row["query"]), str(row["candidate"]), row["signal"]): row
               for _, row in scored.iterrows()}
    result = []
    for record in originals:
        key = (str(record["query_entity_id"]), str(record["candidate_entity_id"]))
        record["original_similarity_score"] = record["similarity_score"]
        score = totals.loc[key, "score"]
        record["similarity_score"] = None if pd.isna(score) else float(score)
        record["coverage"] = float(totals.loc[key, "coverage"])
        for component in record["components"]:
            details = lookups[(*key, component["signal_id"])]
            component["experimental_score"] = float(details["experimental_score"])
            component["experimental_weight"] = float(details["experimental_weight"])
            component["experimental_confidence"] = float(details["usable_confidence"])
            component["experimental_imputed"] = bool(details["imputed"])
        result.append(record)
    return sorted(result, key=lambda row: (
        row["similarity_score"] is None,
        -(row["similarity_score"] if row["similarity_score"] is not None else 0.0),
        str(row["candidate_entity_id"]),
    ))
