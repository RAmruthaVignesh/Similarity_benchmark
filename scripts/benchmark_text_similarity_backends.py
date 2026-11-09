#!/usr/bin/env python3
"""Benchmark the E5 text-signal embedding backend on local RelAmazon.

The benchmark reports two complementary facts: materialization throughput and
the spread of pairwise similarities.  Spread is a diagnostic for representation
collapse, not a relevance metric: semantic retrieval still needs task-specific
evaluation once relevance labels for entity pairs exist.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402
from src.similarity.signal_measures import SignalMeasureExecutor  # noqa: E402


DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"
DEFAULT_CACHE_ROOT = Path("/storage/lcostes/model-cache/huggingface/hub")
DEFAULT_SIGNAL_IDS = (
    "attribute:product:product:product.description",
    "attribute:product:product>review:review.review_text",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs <= 0 or args.embedding_batch_size <= 0:
        parser.error("--pairs and --embedding-batch-size must be positive")

    dataset = LocalRelBenchDataset(args.dataset_root)
    graph = SchemaGraph.from_manifest(dataset.schema)
    catalog = {candidate.id: candidate for candidate in enumerate_signal_candidates(dataset, graph, max_hops=2)}
    missing = [signal_id for signal_id in DEFAULT_SIGNAL_IDS if signal_id not in catalog]
    if missing:
        raise RuntimeError(f"benchmark signals are absent from this catalog: {missing}")
    rng = np.random.default_rng(args.seed)
    results: list[dict[str, object]] = []

    executor = SignalMeasureExecutor(
        dataset,
        graph,
        model_cache_root=args.model_cache_root,
        embedding_batch_size=args.embedding_batch_size,
    )
    for signal_id in DEFAULT_SIGNAL_IDS:
        started = time.perf_counter()
        materialized = executor.materialize(catalog[signal_id])
        elapsed = time.perf_counter() - started
        entity_ids = np.asarray(materialized.entity_ids, dtype=object)
        left_indices = rng.integers(0, len(entity_ids), size=args.pairs)
        right_indices = rng.integers(0, len(entity_ids), size=args.pairs)
        scores = np.asarray(
            [
                executor.score(materialized, entity_ids[left], entity_ids[right])
                for left, right in zip(left_indices, right_indices, strict=True)
                if left != right
            ],
            dtype=float,
        )
        finite = scores[np.isfinite(scores)]
        result = {
            "backend": "e5-vllm",
            "signal_id": signal_id,
            "measure": materialized.measure,
            "entities_encoded": len(materialized.embeddings_by_entity or {}),
            "path_values_encoded": sum(len(values) for values in materialized.values_by_entity.values()),
            "materialization_seconds": round(elapsed, 3),
            "path_values_per_second": round(sum(len(values) for values in materialized.values_by_entity.values()) / elapsed, 2),
            "sampled_nonself_pairs": int(len(finite)),
            "mean_similarity": round(float(np.mean(finite)), 4),
            "std_similarity": round(float(np.std(finite)), 4),
            "p05_similarity": round(float(np.percentile(finite, 5)), 4),
            "p50_similarity": round(float(np.percentile(finite, 50)), 4),
            "p95_similarity": round(float(np.percentile(finite, 95)), 4),
            "share_similarity_ge_0_95": round(float(np.mean(finite >= 0.95)), 4),
        }
        if signal_id == "attribute:product:product:product.description":
            result.update(_category_neighbour_diagnostic(dataset, materialized))
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results}, indent=2) + "\n")
    return 0


def _category_neighbour_diagnostic(dataset: LocalRelBenchDataset, materialized) -> dict[str, float]:
    """Weak intrinsic check: do text-nearest products share specific categories?

    This is deliberately a diagnostic rather than a benchmark target: book
    categories are not the same thing as condition-specific similarity.  The
    comparison with random product pairs nevertheless checks whether an
    embedding space retains meaningful topical structure.
    """
    frame = dataset.get_table("product").to_pandas()
    pkey = dataset.get_primary_key("product")
    assert pkey is not None
    categories_by_id = {
        row[pkey]: frozenset(str(value).casefold() for value in _flatten(row["category"]))
        for _, row in frame.iterrows()
    }
    entities = [entity_id for entity_id in materialized.entity_ids if entity_id in materialized.embeddings_by_entity]
    categories = [categories_by_id.get(entity_id, frozenset()) for entity_id in entities]
    frequency: dict[str, int] = {}
    for values in categories:
        for value in values:
            frequency[value] = frequency.get(value, 0) + 1
    weights = {value: np.log((len(categories) + 1) / (count + 1)) + 1 for value, count in frequency.items()}
    vectors = np.vstack([materialized.embeddings_by_entity[entity_id] for entity_id in entities])
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    top_indices = np.argpartition(similarities, -10, axis=1)[:, -10:]
    top10 = [
        _weighted_jaccard(categories[index], categories[neighbour], weights)
        for index, neighbours in enumerate(top_indices)
        for neighbour in neighbours
    ]
    random_pairs = [
        _weighted_jaccard(categories[left], categories[right], weights)
        for left in range(len(categories))
        for right in range(left + 1, len(categories))
    ]
    return {
        "top10_specific_category_similarity": round(float(np.mean(top10)), 4),
        "random_specific_category_similarity": round(float(np.mean(random_pairs)), 4),
        "top10_category_uplift_over_random": round(float(np.mean(top10) / max(np.mean(random_pairs), 1e-12)), 3),
    }


def _flatten(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(item for nested in value for item in _flatten(nested))
    return (value,)


def _weighted_jaccard(left: frozenset[str], right: frozenset[str], weights: dict[str, float]) -> float:
    union = left | right
    if not union:
        return 0.0
    return sum(weights[value] for value in left & right) / sum(weights[value] for value in union)


if __name__ == "__main__":
    raise SystemExit(main())
