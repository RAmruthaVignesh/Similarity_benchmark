"""Tests for dev-subset sampling, offline. Run: python -m unittest discover tests"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dev_subset import DevSubsetConfig  # noqa: E402
from src.datasets.dev_subset_builder import (  # noqa: E402
    _connected_selection,
    _evenly_spaced,
    build_dev_subset,
)
from src.datasets.dev_subset_validation import validate_dev_subset  # noqa: E402

REL_AMAZON = REPO_ROOT / "data" / "manifests" / "rel-amazon" / "manifest.yaml"

# A time-sorted source review table of 10 row groups x 100 rows, like the real one.
GROUP_ROWS = 100
GROUPS = 10
SOURCE_ROWS = GROUP_ROWS * GROUPS

# 1 prints which source row groups each build sampled from, 0 stays quiet.
VERBOSE = 1


def product_of(position: int) -> int:
    """7 popular products on even rows, 61 rare ones (3-5 rows per 5 groups) on odd."""
    return position % 7 if position % 2 == 0 else 7 + position % 61


def pool_counts(row_groups: int) -> dict[int, int]:
    """Reviews per product within the row groups a build samples from."""
    groups = set(_evenly_spaced(GROUPS, row_groups))
    counts: dict[int, int] = {}
    for position in range(SOURCE_ROWS):
        if position // GROUP_ROWS in groups:
            product = product_of(position)
            counts[product] = counts.get(product, 0) + 1
    return counts


def write_source(root: Path, customer_of=lambda position: position % 37) -> Path:
    """A local stand-in for the remote RelBench layout; returns its base URL."""
    db = root / "source" / "rel-amazon" / "db"
    db.mkdir(parents=True)
    start = datetime(2008, 1, 1)
    pq.write_table(
        pa.table(
            {
                "position": list(range(SOURCE_ROWS)),
                "review_time": [start + timedelta(days=i) for i in range(SOURCE_ROWS)],
                "customer_id": [customer_of(i) for i in range(SOURCE_ROWS)],
                "product_id": [product_of(i) for i in range(SOURCE_ROWS)],
            }
        ),
        db / "review.parquet",
        row_group_size=GROUP_ROWS,
    )
    # More products and customers than reviews reference, so filtering shows.
    pq.write_table(pa.table({"product_id": list(range(80))}), db / "product.parquet")
    pq.write_table(
        pa.table({"customer_id": list(range(SOURCE_ROWS + 50))}),
        db / "customer.parquet",
    )
    return root / "source"


def per_product(review: pa.Table) -> dict[int, int]:
    """Kept reviews per product in a built subset."""
    counts: dict[int, int] = {}
    for product in review["product_id"].to_pylist():
        counts[product] = counts.get(product, 0) + 1
    return dict(sorted(counts.items()))


class TestEvenlySpaced(unittest.TestCase):
    def test_spreads_across_both_ends(self):
        self.assertEqual(_evenly_spaced(20, 10), [0, 2, 4, 6, 8, 11, 13, 15, 17, 19])

    def test_takes_everything_when_asked_for_at_least_all(self):
        self.assertEqual(_evenly_spaced(20, 20), list(range(20)))
        self.assertEqual(_evenly_spaced(20, 30), list(range(20)))

    def test_a_single_group_is_the_middle_one(self):
        self.assertEqual(_evenly_spaced(20, 1), [9])


class TestSampling(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.base_url = str(write_source(self.root))

    SETTINGS = {
        "num_products": 10,
        "min_reviews_per_product": 4,
        "max_reviews_per_product": 6,
        "review_row_groups": 5,
    }

    def build(self, name: str, **overrides) -> tuple[DevSubsetConfig, pa.Table]:
        settings = {**self.SETTINGS, **overrides}
        config = DevSubsetConfig(output_dir=self.root / name, **settings)
        build_dev_subset(config, REL_AMAZON, base_url=self.base_url)
        review = pq.read_table(config.table_path("review"))
        if VERBOSE:
            kept = per_product(review)
            print(
                f"\n  {self.id().rsplit('.', 1)[-1]} {settings}: "
                f"{review.num_rows} rows, products -> kept reviews {kept}"
            )
            sys.stdout.flush()
        return config, review

    def test_the_source_exercises_every_rule(self):
        counts = pool_counts(self.SETTINGS["review_row_groups"]).values()
        low = self.SETTINGS["min_reviews_per_product"]
        high = self.SETTINGS["max_reviews_per_product"]
        self.assertTrue(any(n < low for n in counts), "nothing below the minimum")
        self.assertTrue(any(n > high for n in counts), "nothing to truncate")
        eligible = sum(n >= low for n in counts)
        self.assertGreater(eligible, self.SETTINGS["num_products"])

    def test_selected_products_meet_the_minimum_before_truncation(self):
        _, review = self.build("subset")
        counts = pool_counts(self.SETTINGS["review_row_groups"])
        for product in per_product(review):
            self.assertGreaterEqual(
                counts[product], self.SETTINGS["min_reviews_per_product"]
            )

    def test_no_product_keeps_more_than_the_maximum(self):
        _, review = self.build("subset")
        counts = pool_counts(self.SETTINGS["review_row_groups"])
        high = self.SETTINGS["max_reviews_per_product"]
        for product, kept in per_product(review).items():
            self.assertEqual(kept, min(counts[product], high))

    def test_selects_at_most_num_products(self):
        _, review = self.build("subset")
        self.assertEqual(len(per_product(review)), self.SETTINGS["num_products"])
        _, few = self.build("few", num_products=3)
        self.assertEqual(len(per_product(few)), 3)

    def test_selects_every_eligible_product_when_there_are_fewer(self):
        _, review = self.build("subset", num_products=1_000)
        low = self.SETTINGS["min_reviews_per_product"]
        counts = pool_counts(self.SETTINGS["review_row_groups"])
        self.assertEqual(
            set(per_product(review)), {p for p, n in counts.items() if n >= low}
        )

    def test_draws_only_from_evenly_spaced_row_groups(self):
        _, review = self.build("subset")
        groups = {p // GROUP_ROWS for p in review["position"].to_pylist()}
        self.assertLessEqual(groups, set(_evenly_spaced(GROUPS, 5)))

    def test_is_reproducible(self):
        _, first = self.build("first")
        _, second = self.build("second")
        self.assertTrue(first.equals(second))

    def test_the_seed_changes_the_sample(self):
        _, seed0 = self.build("seed0", seed=0)
        _, seed1 = self.build("seed1", seed=1)
        self.assertNotEqual(set(per_product(seed0)), set(per_product(seed1)))

    def test_rows_keep_source_order(self):
        _, review = self.build("subset")
        positions = review["position"].to_pylist()
        self.assertEqual(positions, sorted(positions))

    def test_referenced_tables_stay_closed_and_minimal(self):
        config, review = self.build("subset")
        report = validate_dev_subset(config)
        self.assertTrue(report.ok, report.failures)
        product = pq.read_table(config.table_path("product"))
        self.assertEqual(
            set(product["product_id"].to_pylist()),
            set(review["product_id"].to_pylist()),
        )


class TestConnectedSelection(unittest.TestCase):
    def test_expands_product_to_customer_to_product(self):
        # a -c1- b -c2- c is a chain; d comes earlier in seed order but is isolated.
        links = [("a", "c1"), ("b", "c1"), ("b", "c2"), ("c", "c2")]
        self.assertEqual(
            _connected_selection(["a", "d", "b", "c"], links, 3), ["a", "b", "c"]
        )

    def test_prefers_products_sharing_more_customers(self):
        links = [("a", "c1"), ("a", "c2"), ("a", "c3"), ("b", "c1"), ("c", "c2"), ("c", "c3")]
        self.assertEqual(
            _connected_selection(["a", "b", "c"], links, 3), ["a", "c", "b"]
        )

    def test_falls_back_to_the_next_seed_when_a_component_runs_out(self):
        links = [("a", "c1"), ("b", "c1"), ("c", "c2"), ("d", "c2")]
        order = ["a", "c", "b", "d"]
        self.assertEqual(_connected_selection(order, links, 3), ["a", "b", "c"])
        self.assertEqual(_connected_selection(order, links, 10), ["a", "b", "c", "d"])

    def test_never_picks_or_walks_through_ineligible_products(self):
        # x would bridge a and b, but it is not eligible.
        links = [("a", "c1"), ("x", "c1"), ("x", "c2"), ("b", "c2")]
        self.assertEqual(_connected_selection(["a", "b"], links, 5), ["a", "b"])

    def test_is_deterministic(self):
        links = [(p, c) for p in range(30) for c in range(p % 4, 40, 7)]
        order = list(range(29, -1, -1))
        self.assertEqual(
            _connected_selection(order, links, 12),
            _connected_selection(order, list(reversed(links)), 12),
        )


class TestOverlapAwareSampling(TestSampling):
    """Every entity-centred guarantee, plus connectivity, in overlap-aware mode."""

    SETTINGS = {**TestSampling.SETTINGS, "sampling_mode": "overlap_aware"}

    def test_every_selected_product_shares_a_customer(self):
        config, _ = self.build("subset")
        overlap = validate_dev_subset(config).overlap
        self.assertEqual(overlap.connected_entities, overlap.entities)

    def test_overlaps_more_than_entity_centered_sampling(self):
        connected, _ = self.build("connected")
        spread, _ = self.build("spread", sampling_mode="entity_centered")
        self.assertGreater(
            validate_dev_subset(connected).overlap.overlapping_pairs,
            validate_dev_subset(spread).overlap.overlapping_pairs,
        )

    def test_falls_back_to_new_seeds_when_no_customer_is_shared(self):
        self.base_url = str(
            write_source(self.root / "isolated", customer_of=lambda position: position)
        )
        config, review = self.build("isolated")
        self.assertEqual(len(per_product(review)), self.SETTINGS["num_products"])
        report = validate_dev_subset(config)
        self.assertTrue(report.ok, report.failures)
        self.assertEqual(report.overlap.overlapping_pairs, 0)


if __name__ == "__main__":
    unittest.main()
