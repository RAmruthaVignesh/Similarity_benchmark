"""Tests for deterministic signal materialization and signal-specific measures."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import SignalCandidate  # noqa: E402
from src.similarity.signal_measures import SignalMeasureExecutor, combine_signal_scores  # noqa: E402


class SignalMeasureTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "manifest.yaml").write_text(
            """
name: measure-test
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
                    "customer_id": [1, 1, 2, 2, 3],
                    "product_id": [10, 11, 11, 12, 12],
                    "rating": [5, 4, 4, 2, 5],
                    "verified": [True, False, True, False, True],
                    "review_time": [
                        "2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"
                    ],
                    "review_text": ["bright lamp", "warm light", "warm light", "dim light", "dim light"],
                }
            ),
            root / "review.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "product_id": [10, 11, 12],
                    "price": [10.0, 20.0, 30.0],
                    "category": [["lighting", "lamp"], ["lighting", "lamp"], ["appliance", "fan"]],
                    "description": ["bright wireless lamp", "wireless warm light", "quiet cooling fan"],
                }
            ),
            root / "product.parquet",
        )
        pq.write_table(
            pa.table({"customer_id": [1, 2, 3], "customer_name": ["Alice", "Alicia", "Bob"]}),
            root / "customer.parquet",
        )
        dataset = LocalRelBenchDataset(root)
        self.executor = SignalMeasureExecutor(dataset, SchemaGraph.from_manifest(dataset.schema))

    def test_direct_numeric_attribute_uses_percentile_distance(self):
        signal = SignalCandidate(
            id="attribute:product:product:product.price", kind="attribute", source_table="product",
            path=("product",), attribute_table="product", attribute_column="price",
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "percentile_distance")
        self.assertAlmostEqual(self.executor.score(materialized, 10, 11), 2 / 3)
        self.assertEqual(self.executor.score(materialized, 10, 10), 1.0)
        only_one = self.executor.score_query(10, [materialized], candidate_entity_ids=["10", "12"])
        self.assertEqual([result.candidate_entity_id for result in only_one], [12])
        with self.assertRaises(ValueError):
            self.executor.score_query(10, [materialized], candidate_entity_ids=["unknown"])

    def test_text_attribute_uses_e5_embedding_cosine(self):
        signal = SignalCandidate(
            id="attribute:product:product:product.description", kind="attribute", source_table="product",
            path=("product",), attribute_table="product", attribute_column="description",
        )
        embeddings = {
            10: np.array([1.0, 0.0], dtype=np.float32),
            11: np.array([0.9, 0.1], dtype=np.float32),
            12: np.array([0.0, 1.0], dtype=np.float32),
        }
        with patch.object(self.executor, "_embed_e5_text_collections", return_value=embeddings):
            materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "e5_embedding_cosine")
        self.assertGreater(self.executor.score(materialized, 10, 11), self.executor.score(materialized, 10, 12))

    def test_manifest_time_column_uses_temporal_percentile_distance_even_if_stored_as_text(self):
        signal = SignalCandidate(
            id="attribute:review:review:review.review_time", kind="attribute", source_table="review",
            path=("review",), attribute_table="review", attribute_column="review_time",
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "temporal_percentile_distance")
        self.assertGreater(self.executor.score(materialized, 0, 1), self.executor.score(materialized, 0, 4))

    def test_relational_path_materializes_customer_product_sets_and_jaccard(self):
        signal = SignalCandidate(
            id="relationship:customer:customer>review>product", kind="relationship", source_table="customer",
            path=("customer", "review", "product"),
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "jaccard_neighborhood")
        self.assertEqual(set(materialized.values_for(1)), {10, 11})
        self.assertEqual(set(materialized.values_for(2)), {11, 12})
        self.assertAlmostEqual(self.executor.score(materialized, 1, 2), 1 / 3)

    def test_relationship_without_target_key_falls_back_to_degree_similarity(self):
        signal = SignalCandidate(
            id="relationship:product:product>review", kind="relationship", source_table="product",
            path=("product", "review"),
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "degree_similarity")
        self.assertEqual(self.executor.score(materialized, 10, 11), 0.5)

    def test_list_categories_are_flattened_and_use_idf_weighted_jaccard(self):
        signal = SignalCandidate(
            id="attribute:product:product:product.category", kind="attribute", source_table="product",
            path=("product",), attribute_table="product", attribute_column="category",
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "idf_weighted_jaccard")
        self.assertEqual(set(materialized.values_for(10)), {"lighting", "lamp"})
        self.assertEqual(self.executor.score(materialized, 10, 11), 1.0)
        self.assertEqual(self.executor.score(materialized, 10, 12), 0.0)

    def test_path_aggregated_booleans_use_rate_similarity(self):
        signal = SignalCandidate(
            id="attribute:product:product>review:review.verified", kind="attribute", source_table="product",
            path=("product", "review"), attribute_table="review", attribute_column="verified",
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.measure, "boolean_rate_similarity")
        self.assertAlmostEqual(self.executor.score(materialized, 10, 11), 0.5)

    def test_entity_names_are_categorical_not_semantic_text(self):
        signal = SignalCandidate(
            id="attribute:customer:customer:customer.customer_name", kind="attribute", source_table="customer",
            path=("customer",), attribute_table="customer", attribute_column="customer_name",
        )
        materialized = self.executor.materialize(signal)
        self.assertEqual(materialized.value_kind, "categorical")
        self.assertEqual(materialized.measure, "categorical_agreement")

    def test_weighted_combination_skips_missing_components(self):
        self.assertAlmostEqual(
            combine_signal_scores([
                {"score": 0.2, "weight": 1.0},
                {"score": 0.8, "weight": 3.0},
                {"score": None, "weight": 100.0},
            ]),
            0.65,
        )


if __name__ == "__main__":
    unittest.main()
