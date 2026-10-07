#!/usr/bin/env python3
"""Validate a local development subset and print its basic statistics."""

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
    DEV_ROOT,
    DevSubsetConfig,
)
from src.datasets.dev_subset_validation import (  # noqa: E402
    SubsetReport,
    validate_dev_subset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--num-products",
        type=int,
        default=DEFAULT_NUM_PRODUCTS,
        help="value the subset was built with; bounds its review count",
    )
    parser.add_argument(
        "--min-reviews-per-product",
        type=int,
        default=DEFAULT_MIN_REVIEWS_PER_PRODUCT,
    )
    parser.add_argument(
        "--max-reviews-per-product",
        type=int,
        default=DEFAULT_MAX_REVIEWS_PER_PRODUCT,
        help="value the subset was built with; bounds its review count",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"subset directory (default: {DEV_ROOT}/<dataset>)",
    )
    args = parser.parse_args()

    try:
        config = DevSubsetConfig(
            dataset=args.dataset,
            num_products=args.num_products,
            min_reviews_per_product=args.min_reviews_per_product,
            max_reviews_per_product=args.max_reviews_per_product,
            output_dir=args.output_dir or REPO_ROOT / DEV_ROOT / args.dataset,
        )
        report = validate_dev_subset(config)
    except (OSError, ValueError, duckdb.Error) as error:
        parser.exit(2, f"error: {error}\n")

    print_report(config, report)
    return 0 if report.ok else 1


def print_report(config: DevSubsetConfig, report: SubsetReport) -> None:
    print(f"{report.dataset} subset in {config.output_dir}/")

    print("\nstatistics")
    if report.fact_table in report.row_counts:
        print(f"  {report.fact_table + ' rows':<28} {report.row_counts[report.fact_table]:>10,}")
    for table, count in report.distinct_keys.items():
        print(f"  {'unique ' + table + 's':<28} {count:>10,}")
    for table, summary in report.rows_per_key.items():
        per = f"{report.fact_table}s per {table}"
        print(f"  {'min ' + per:<28} {summary.min:>10,}")
        print(f"  {'median ' + per:<28} {summary.median:>10,.1f}")
        print(f"  {'mean ' + per:<28} {summary.mean:>10,.1f}")
        print(f"  {'max ' + per:<28} {summary.max:>10,}")
    if report.time_range is not None:
        earliest, latest = report.time_range
        print(f"  {report.time_column + ' range':<28} {earliest} .. {latest}")

    overlap = report.overlap
    if overlap is not None:
        entities, links = f"{overlap.entity_table}s", f"{overlap.link_table}s"
        print(f"\n{overlap.entity_table} overlap through shared {links}")
        print(
            f"  {entities + ' sharing a ' + overlap.link_table:<36} "
            f"{overlap.connected_entities:>8,} of {overlap.entities:,} "
            f"({overlap.connected_entities / max(overlap.entities, 1):.1%})"
        )
        print(
            f"  {'pairs with nonzero overlap':<36} {overlap.overlapping_pairs:>8,} "
            f"of {overlap.possible_pairs:,} ({overlap.overlapping_pair_fraction:.2%})"
        )
        if overlap.overlapping_pairs:
            print(
                f"  {'shared ' + links + ' median/mean/max':<36} "
                f"{overlap.shared_median:>8.1f} / {overlap.shared_mean:.2f} / "
                f"{overlap.shared_max:,}"
            )
            print(f"  {'median nonzero Jaccard':<36} {overlap.median_jaccard:>8.4f}")
        print(
            f"  {links + ' linking 2+ ' + entities:<36} {overlap.bridging_links:>8,}"
        )

    print("\nchecks")
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        detail = f"  ({check.detail})" if check.detail else ""
        print(f"  {status}  {check.name}{detail}")

    failures = len(report.failures)
    print("\nOK" if not failures else f"\nFAILED: {failures} check(s)")


if __name__ == "__main__":
    raise SystemExit(main())
