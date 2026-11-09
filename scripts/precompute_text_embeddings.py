#!/usr/bin/env python3
"""Precompute and persist semantic text embeddings for a local RelBench dataset.

Embeddings are cached by model snapshot and normalized, truncated E5 input, so
later similarity runs can reuse text vectors across signals and path views.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.similarity.signal_measures import (  # noqa: E402
    DEFAULT_E5_MODEL,
    DEFAULT_MODEL_CACHE_ROOT,
    DEFAULT_TEXT_EMBEDDING_CACHE,
    SignalMeasureExecutor,
)

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "dev" / "rel-amazon"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--text-embedding-model", default=DEFAULT_E5_MODEL)
    parser.add_argument("--model-cache-root", type=Path, default=DEFAULT_MODEL_CACHE_ROOT)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=DEFAULT_TEXT_EMBEDDING_CACHE,
        help="persistent SQLite cache file (default: /storage/lcostes/simbench-cache/text-embeddings.sqlite3)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    temporary_root = Path(os.environ.get("SIMBENCH_TMPDIR", "/storage/lcostes/tmp"))
    temporary_root.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(temporary_root)

    try:
        dataset = LocalRelBenchDataset(args.dataset_root)
        graph = SchemaGraph.from_manifest(dataset.schema)
        executor = SignalMeasureExecutor(
            dataset,
            graph,
            text_embedding_model=args.text_embedding_model,
            model_cache_root=args.model_cache_root,
            text_embedding_cache=args.embedding_cache,
            embedding_batch_size=args.batch_size,
        )
        report = executor.precompute_text_embeddings()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
