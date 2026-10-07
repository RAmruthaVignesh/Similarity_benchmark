"""Tests for the development-subset config. Run: python -m unittest discover tests"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.dev_subset import DevSubsetConfig, DevSubsetError  # noqa: E402
from src.datasets.relbench_manifest import load_manifest  # noqa: E402

REL_AMAZON = REPO_ROOT / "data" / "manifests" / "rel-amazon" / "manifest.yaml"

# 1 prints the default subset layout when the tests run, 0 stays quiet.
VERBOSE = 1


def setUpModule() -> None:
    if VERBOSE:
        print_report(DevSubsetConfig())


def print_report(config: DevSubsetConfig) -> None:
    """The planned subset layout, for eyeballing during development."""
    print("\ndev subset config")
    for key, value in config.to_dict().items():
        print(f"  {key:14s} {value}")

    print(f"\nlayout under {config.output_dir}/")
    tables = load_manifest(REL_AMAZON).table_names
    for path in [*map(config.table_path, tables), config.manifest_path]:
        status = "present" if (REPO_ROOT / path).exists() else "missing"
        print(f"  {status:8s} {path}")
    print()
    sys.stdout.flush()


class TestDefaults(unittest.TestCase):
    def test_documented_defaults(self):
        config = DevSubsetConfig()
        self.assertEqual(config.dataset, "rel-amazon")
        self.assertEqual(config.num_products, 500)
        self.assertEqual(config.min_reviews_per_product, 20)
        self.assertEqual(config.max_reviews_per_product, 100)
        self.assertEqual(config.sampling_mode, "entity_centered")
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.output_dir, Path("data/dev/rel-amazon"))

    def test_output_dir_follows_the_dataset_name(self):
        self.assertEqual(
            DevSubsetConfig(dataset="rel-stack").output_dir,
            Path("data/dev/rel-stack"),
        )

    def test_output_dir_can_be_overridden_with_a_string_or_path(self):
        self.assertEqual(
            DevSubsetConfig(output_dir="/tmp/subset").output_dir, Path("/tmp/subset")
        )
        self.assertEqual(
            DevSubsetConfig(output_dir=Path("/tmp/subset")).output_dir,
            Path("/tmp/subset"),
        )


class TestPaths(unittest.TestCase):
    def test_paths_match_the_intended_layout(self):
        config = DevSubsetConfig()
        self.assertEqual(
            [config.table_path(t) for t in ("review", "product", "customer")],
            [
                Path("data/dev/rel-amazon/review.parquet"),
                Path("data/dev/rel-amazon/product.parquet"),
                Path("data/dev/rel-amazon/customer.parquet"),
            ],
        )
        self.assertEqual(
            config.manifest_path, Path("data/dev/rel-amazon/manifest.yaml")
        )

    def test_paths_cover_every_table_the_manifest_declares(self):
        config = DevSubsetConfig()
        for table in load_manifest(REL_AMAZON).table_names:
            self.assertEqual(config.table_path(table).parent, config.output_dir)

    def test_constructing_a_config_creates_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = DevSubsetConfig(output_dir=Path(tmp) / "subset")
            config.manifest_path
            config.table_path("review")
            self.assertFalse(config.output_dir.exists())


class TestValidation(unittest.TestCase):
    def test_review_rows_is_the_most_the_subset_can_hold(self):
        config = DevSubsetConfig(num_products=7, max_reviews_per_product=30)
        self.assertEqual(config.review_rows, 210)

    def test_counts_must_be_positive(self):
        for field in ("num_products", "min_reviews_per_product"):
            for value in (0, -1):
                with self.assertRaises(DevSubsetError):
                    DevSubsetConfig(**{field: value})

    def test_max_reviews_cannot_be_below_min_reviews(self):
        with self.assertRaises(DevSubsetError):
            DevSubsetConfig(min_reviews_per_product=20, max_reviews_per_product=19)
        DevSubsetConfig(min_reviews_per_product=20, max_reviews_per_product=20)

    def test_sampling_mode_must_be_known(self):
        for mode in ("entity_centered", "overlap_aware"):
            self.assertEqual(DevSubsetConfig(sampling_mode=mode).sampling_mode, mode)
        with self.assertRaises(DevSubsetError):
            DevSubsetConfig(sampling_mode="random")

    def test_dataset_must_not_be_empty(self):
        with self.assertRaises(DevSubsetError):
            DevSubsetConfig(dataset="")

    def test_config_is_frozen_but_replaceable(self):
        config = DevSubsetConfig()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.num_products = 5
        smaller = dataclasses.replace(config, num_products=100)
        self.assertEqual(smaller.num_products, 100)
        self.assertEqual(smaller.output_dir, config.output_dir)

    def test_to_dict_is_json_serializable(self):
        self.assertEqual(
            json.loads(json.dumps(DevSubsetConfig().to_dict()))["output_dir"],
            "data/dev/rel-amazon",
        )


if __name__ == "__main__":
    unittest.main()
