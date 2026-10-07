"""Tests for dev-subset validation. Run: python -m unittest discover tests"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dev_subset import DevSubsetConfig  # noqa: E402
from src.datasets.dev_subset_validation import (  # noqa: E402
    CountSummary,
    OverlapSummary,
    SubsetReport,
    validate_dev_subset,
)

REL_AMAZON = REPO_ROOT / "data" / "manifests" / "rel-amazon" / "manifest.yaml"

MISSING = object()

# 1 prints each test's validation report as it runs, 0 stays quiet.
VERBOSE = 1


def print_report(test_name: str, report: SubsetReport) -> None:
    """One test's validation outcome, for eyeballing during development."""
    print(f"\n{test_name}")
    rows = ", ".join(f"{t}={n}" for t, n in report.row_counts.items()) or "-"
    print(f"  rows:   {rows}")
    unique = ", ".join(f"{t}={n}" for t, n in report.distinct_keys.items()) or "-"
    print(f"  unique: {unique}")
    if report.time_range is not None:
        earliest, latest = report.time_range
        print(f"  {report.time_column}: {earliest} .. {latest}")
    if report.ok:
        print(f"  all {len(report.checks)} checks pass")
    for check in report.failures:
        detail = f"  ({check.detail})" if check.detail else ""
        print(f"  FAIL {check.name}{detail}")
    sys.stdout.flush()

# A tiny valid rel-amazon subset; the null product_id is allowed by the checks.
REVIEW = {
    "review_time": [datetime(2015, 1, 1), datetime(2015, 6, 1), datetime(2016, 2, 1)],
    "customer_id": [1, 1, 2],
    "product_id": [10, 11, None],
    "rating": [5.0, 4.0, 3.0],
}
PRODUCT = {"product_id": [10, 11], "title": ["lamp", "mug"]}
CUSTOMER = {"customer_id": [1, 2], "name": ["ada", "bo"]}


class SubsetValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name) / "rel-amazon"

    def validate(
        self,
        review=REVIEW,
        product=PRODUCT,
        customer=CUSTOMER,
        review_rows: int = 3,
        manifest: bool = True,
    ) -> SubsetReport:
        config = DevSubsetConfig(
            num_products=1,
            min_reviews_per_product=1,
            max_reviews_per_product=review_rows,
            output_dir=self.root,
        )
        self.root.mkdir(parents=True)
        for name, columns in (
            ("review", review),
            ("product", product),
            ("customer", customer),
        ):
            if columns is not MISSING:
                pq.write_table(pa.table(columns), config.table_path(name))
        if manifest:
            shutil.copyfile(REL_AMAZON, config.manifest_path)
        report = validate_dev_subset(config)
        if VERBOSE:
            print_report(self.id().rsplit(".", 1)[-1], report)
        return report

    def check(self, report: SubsetReport, name: str):
        matches = [check for check in report.checks if check.name == name]
        self.assertEqual(len(matches), 1, f"no single check named {name!r}")
        return matches[0]

    def assertFailsOnly(self, report: SubsetReport, name: str) -> None:
        self.assertEqual([check.name for check in report.failures], [name])


class TestValidSubset(SubsetValidationTest):
    def test_a_valid_subset_passes_every_check(self):
        report = self.validate()
        self.assertTrue(report.ok, report.failures)

    def test_statistics(self):
        report = self.validate()
        self.assertEqual(report.fact_table, "review")
        self.assertEqual(
            report.row_counts, {"review": 3, "product": 2, "customer": 2}
        )
        self.assertEqual(report.distinct_keys, {"product": 2, "customer": 2})
        self.assertEqual(
            report.rows_per_key,
            {
                "customer": CountSummary(min=1, median=1.5, mean=1.5, max=2),
                "product": CountSummary(min=1, median=1.0, mean=1.0, max=1),
            },
        )
        # Customer 1 reviewed both products: one pair, one shared customer.
        self.assertEqual(
            report.overlap,
            OverlapSummary(
                entity_table="product",
                link_table="customer",
                entities=2,
                connected_entities=2,
                overlapping_pairs=1,
                bridging_links=1,
                shared_median=1.0,
                shared_mean=1.0,
                shared_max=1,
                median_jaccard=1.0,
            ),
        )
        self.assertEqual(report.overlap.overlapping_pair_fraction, 1.0)
        self.assertEqual(report.time_column, "review_time")
        self.assertEqual(
            report.time_range, (datetime(2015, 1, 1), datetime(2016, 2, 1))
        )

    def test_both_foreign_keys_are_checked(self):
        report = self.validate()
        for name in (
            "review.customer_id -> customer.customer_id",
            "review.product_id -> product.product_id",
        ):
            self.assertTrue(self.check(report, name).passed)

    def test_null_foreign_keys_are_not_dangling(self):
        report = self.validate(review={**REVIEW, "product_id": [None, None, None]})
        self.assertTrue(
            self.check(report, "review.product_id -> product.product_id").passed
        )


class TestFailures(SubsetValidationTest):
    def test_missing_manifest(self):
        report = self.validate(manifest=False)
        self.assertFailsOnly(report, "manifest exists")

    def test_missing_table(self):
        report = self.validate(customer=MISSING)
        self.assertFailsOnly(report, "table customer exists")
        self.assertNotIn("customer", report.row_counts)

    def test_empty_table(self):
        empty = {"product_id": pa.array([], pa.int64()), "title": pa.array([], pa.string())}
        report = self.validate(product=empty)
        self.assertIn("table product is not empty", [c.name for c in report.failures])

    def test_dangling_product_id(self):
        report = self.validate(review={**REVIEW, "product_id": [10, 99, None]})
        self.assertFailsOnly(report, "review.product_id -> product.product_id")
        self.assertEqual(
            self.check(report, "review.product_id -> product.product_id").detail,
            "1 rows without a match",
        )

    def test_dangling_customer_id(self):
        report = self.validate(review={**REVIEW, "customer_id": [1, 7, 8]})
        self.assertFailsOnly(report, "review.customer_id -> customer.customer_id")

    def test_duplicate_product_id(self):
        report = self.validate(product={"product_id": [10, 11, 11], "title": ["a", "b", "c"]})
        self.assertFailsOnly(report, "product.product_id is unique")

    def test_duplicate_customer_id(self):
        report = self.validate(customer={"customer_id": [1, 2, 2], "name": ["a", "b", "c"]})
        self.assertFailsOnly(report, "customer.customer_id is unique")

    def test_more_reviews_than_the_sample_size(self):
        report = self.validate(review_rows=2)
        self.assertFailsOnly(report, "review has at most 2 rows")


if __name__ == "__main__":
    unittest.main()
