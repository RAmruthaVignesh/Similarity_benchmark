#!/usr/bin/env python3
"""Build a local development subset of a RelBench dataset from its remote parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import duckdb  # noqa: E402

from src.datasets.dev_subset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_MAX_REVIEWS_PER_PRODUCT,
    DEFAULT_MIN_REVIEWS_PER_PRODUCT,
    DEFAULT_NUM_PRODUCTS,
    DEFAULT_REVIEW_ROW_GROUPS,
    DEFAULT_SAMPLING_MODE,
    DEFAULT_SEED,
    DEV_ROOT,
    SAMPLING_MODES,
    DevSubsetConfig,
)
from src.datasets.dev_subset_builder import build_dev_subset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--num-products", type=int, default=DEFAULT_NUM_PRODUCTS)
    parser.add_argument(
        "--min-reviews-per-product",
        type=int,
        default=DEFAULT_MIN_REVIEWS_PER_PRODUCT,
    )
    parser.add_argument(
        "--max-reviews-per-product",
        type=int,
        default=DEFAULT_MAX_REVIEWS_PER_PRODUCT,
    )
    parser.add_argument(
        "--sampling-mode",
        choices=SAMPLING_MODES,
        default=DEFAULT_SAMPLING_MODE,
        help="pick products pseudo-randomly, or as a set sharing customers "
        f"(default: {DEFAULT_SAMPLING_MODE})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--review-row-groups",
        type=int,
        default=DEFAULT_REVIEW_ROW_GROUPS,
        help="evenly spaced source row groups to sample from; each one read costs "
        f"its full size in download (default: {DEFAULT_REVIEW_ROW_GROUPS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"where to write the subset (default: {DEV_ROOT}/<dataset>)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="source manifest (default: data/manifests/<dataset>/manifest.yaml)",
    )
    args = parser.parse_args()

    manifest_path = args.manifest or (
        REPO_ROOT / "data" / "manifests" / args.dataset / "manifest.yaml"
    )
    try:
        config = DevSubsetConfig(
            dataset=args.dataset,
            num_products=args.num_products,
            min_reviews_per_product=args.min_reviews_per_product,
            max_reviews_per_product=args.max_reviews_per_product,
            seed=args.seed,
            review_row_groups=args.review_row_groups,
            sampling_mode=args.sampling_mode,
            output_dir=args.output_dir or REPO_ROOT / DEV_ROOT / args.dataset,
        )
        print(f"building {config.dataset} subset in {config.output_dir}/ ...")
        rows = build_dev_subset(config, manifest_path)
    except (OSError, ValueError, duckdb.Error) as error:
        parser.exit(2, f"error: {error}\n")

    for table, count in rows.items():
        print(f"  {count:>8,} rows  {config.table_path(table)}")
    print(f"  manifest        {config.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
