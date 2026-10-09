"""Tests for lazy access to a local manifest-described RelBench subset."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.datasets.relational_dataset import RelationalDataset  # noqa: E402
from src.datasets.relbench_manifest import ForeignKey  # noqa: E402

# 1 prints an interface report, 0 keeps the test output quiet.
VERBOSE = 1


class LocalRelBenchDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "manifest.yaml").write_text(
            """
name: test-dataset
tables:
  review:
    pkey: null
    time_col: review_time
    fkeys:
      customer_id: customer
      product_id: product
  product:
    pkey: product_id
    time_col: null
    fkeys: {}
  customer:
    pkey: customer_id
    time_col: null
    fkeys: {}
""".lstrip()
        )
        pq.write_table(
            pa.table(
                {
                    "review_time": ["2020-01-01", "2020-01-02", "2020-01-03"],
                    "customer_id": [1, 2, 1],
                    "product_id": [10, 10, 11],
                }
            ),
            self.root / "review.parquet",
        )
        pq.write_table(
            pa.table({"product_id": [10, 11], "name": ["lamp", "mug"]}),
            self.root / "product.parquet",
        )
        pq.write_table(
            pa.table({"customer_id": [1, 2]}), self.root / "customer.parquet"
        )

    def test_implements_the_generic_interface(self):
        self.assertIsInstance(LocalRelBenchDataset(self.root), RelationalDataset)

    def test_interface_report(self):
        dataset = LocalRelBenchDataset(self.root)
        if VERBOSE:
            print("\nlocal relational dataset")
            print(f"  tables:              {dataset.get_tables()}")
            print(f"  has product:         {dataset.has_table('product')}")
            print(f"  product primary key: {dataset.get_primary_key('product')}")
            print(f"  review time column:  {dataset.get_time_column('review')}")
            print("  foreign keys:")
            for source, foreign_key in dataset.get_foreign_keys():
                print(
                    f"    {source}.{foreign_key.column} -> "
                    f"{foreign_key.target_table}.{foreign_key.target_column}"
                )
            print(
                "  product 10:          "
                f"{dataset.get_entity('product', 10).to_pylist()[0]}"
            )
            print(
                "  reviews for product 10: "
                f"{dataset.get_rows('review', 'product_id', 10).num_rows}"
            )
            sys.stdout.flush()

    def test_lists_manifest_tables_without_loading_parquet(self):
        with patch(
            "src.datasets.local_relbench_dataset.pq.read_table",
            wraps=pq.read_table,
        ) as read_table:
            dataset = LocalRelBenchDataset(self.root)
            self.assertEqual(dataset.get_tables(), ("review", "product", "customer"))
            self.assertEqual(dataset.get_primary_key("product"), "product_id")
            self.assertEqual(dataset.get_time_column("review"), "review_time")
            read_table.assert_not_called()

    def test_loads_a_table_only_when_requested_and_caches_it(self):
        with patch(
            "src.datasets.local_relbench_dataset.pq.read_table",
            wraps=pq.read_table,
        ) as read_table:
            dataset = LocalRelBenchDataset(self.root)
            product = dataset.get_table("product")
            self.assertEqual(
                product.to_pydict(),
                {"product_id": [10, 11], "name": ["lamp", "mug"]},
            )
            self.assertIs(dataset.get_table("product"), product)
            read_table.assert_called_once_with(self.root / "product.parquet")

    def test_get_entity_uses_the_declared_primary_key(self):
        dataset = LocalRelBenchDataset(self.root)
        self.assertEqual(
            dataset.get_entity("product", 10).to_pydict(),
            {"product_id": [10], "name": ["lamp"]},
        )

    def test_get_rows_filters_one_column(self):
        dataset = LocalRelBenchDataset(self.root)
        self.assertEqual(
            dataset.get_rows("review", "product_id", 10).to_pydict(),
            {
                "review_time": ["2020-01-01", "2020-01-02"],
                "customer_id": [1, 2],
                "product_id": [10, 10],
            },
        )

    def test_has_table(self):
        dataset = LocalRelBenchDataset(self.root)
        self.assertTrue(dataset.has_table("product"))
        self.assertFalse(dataset.has_table("missing"))

    def test_exposes_manifest_foreign_keys(self):
        dataset = LocalRelBenchDataset(self.root)
        self.assertEqual(
            dataset.get_foreign_keys(),
            (
                (
                    "review",
                    ForeignKey("customer_id", "customer", "customer_id"),
                ),
                (
                    "review",
                    ForeignKey("product_id", "product", "product_id"),
                ),
            ),
        )

    def test_returns_none_for_undeclared_optional_columns(self):
        dataset = LocalRelBenchDataset(self.root)
        self.assertIsNone(dataset.get_primary_key("review"))
        self.assertIsNone(dataset.get_time_column("product"))

    def test_unknown_table_is_rejected_before_file_access(self):
        dataset = LocalRelBenchDataset(self.root)
        with self.assertRaisesRegex(KeyError, "unknown table 'missing'"):
            dataset.get_rows("missing", "product_id", 10)

    def test_unknown_column_is_rejected(self):
        dataset = LocalRelBenchDataset(self.root)
        with self.assertRaisesRegex(
            KeyError, "unknown column 'missing' in table 'product'"
        ):
            dataset.get_rows("product", "missing", 10)

    def test_missing_entity_is_rejected(self):
        dataset = LocalRelBenchDataset(self.root)
        with self.assertRaisesRegex(
            KeyError, "table 'product' has no entity with product_id=999"
        ):
            dataset.get_entity("product", 999)


if __name__ == "__main__":
    unittest.main()
