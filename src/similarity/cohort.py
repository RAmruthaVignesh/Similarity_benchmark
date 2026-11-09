"""Resolve explicit entity pools against the frozen dataset's entity IDs."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence


def resolve_entity_ids(requested: Sequence[object], available: Sequence[object]) -> list[object]:
    by_string = {str(value): value for value in available}
    if len(by_string) != len(available):
        raise ValueError("entity IDs are ambiguous when converted to strings")
    if len({str(value) for value in requested}) != len(requested):
        raise ValueError("entity pool contains duplicate IDs")
    missing = [value for value in requested if str(value) not in by_string]
    if missing:
        raise ValueError(f"entity pool contains unknown IDs: {missing[:5]}")
    return [by_string[str(value)] for value in requested]


def load_entity_ids(path: Path) -> list[object]:
    values = json.loads(path.read_text())
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} must contain a nonempty JSON array of entity IDs")
    if any(isinstance(value, (dict, list, bool)) or value is None for value in values):
        raise ValueError(f"{path} contains an invalid entity ID")
    return values


def sample_cohort(ids, table, seed, candidate_count, query_count):
    """Sample a stable pool and a nested reference sample without replacement."""
    if candidate_count < 0 or query_count < 0:
        raise ValueError("sample counts must be non-negative")
    if len(ids) < 2 or len(set(map(str, ids))) != len(ids):
        raise ValueError(f"{table} must contain at least two unique entity IDs")
    table_seed = int.from_bytes(hashlib.sha256(f"{seed}:{table}".encode()).digest()[:8], "big")
    rng = random.Random(table_seed)
    candidates = rng.sample(list(ids), min(candidate_count or len(ids), len(ids)))
    queries = rng.sample(candidates, min(query_count or len(candidates), len(candidates)))
    if len(candidates) < 2:
        raise ValueError("candidate pool must contain at least two entities")
    return {"entity_count": len(ids), "candidate_ids": candidates, "query_ids": queries,
            "entity_id_kind": "row_position" if table == "review" else "primary_key"}
