#!/usr/bin/env python3
"""Build controlled qrels on a schema-only snapshot of real RelAmazon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.synthetic.controlled_signal_benchmark_v2 import (  # noqa: E402
    BENCHMARK_VERSION,
    DEFAULT_SCHEMA_ROOT,
    build_controlled_signal_benchmark,
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "benchmarks" / "controlled-signal-discovery" / "rel-amazon-v2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory to create or regenerate",
    )
    parser.add_argument(
        "--schema-root",
        type=Path,
        default=DEFAULT_SCHEMA_ROOT,
        help="RelAmazon manifest and Parquet files; only metadata is copied",
    )
    args = parser.parse_args()
    tasks = build_controlled_signal_benchmark(
        args.output_root, schema_root=args.schema_root
    )
    print(
        f"Built {BENCHMARK_VERSION}: {len(tasks)} graded tasks with the standard "
        f"39-candidate pool at "
        f"{args.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
