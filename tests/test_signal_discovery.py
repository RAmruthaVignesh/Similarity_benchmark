"""Tests for deterministic schema candidates and local scoring adapters."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.local_relbench_dataset import LocalRelBenchDataset  # noqa: E402
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402
from src.signals.scorers import DeciderSignalScorer  # noqa: E402


class _FakeDecider:
    def decide(self, state, questions):
        del state
        answers = []
        for question in questions:
            options = question["options"]
            answers.append(
                {
                    "choice": options[0],
                    "probs": {
                        option: 0.8 if index == 0 else 0.2 / (len(options) - 1)
                        for index, option in enumerate(options)
                    },
                }
            )
        return answers


class SignalDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "manifest.yaml").write_text(
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
                    "review_time": ["2020-01-01"],
                    "customer_id": [1],
                    "product_id": [10],
                    "rating": [5],
                    "body": ["Useful product"],
                }
            ),
            root / "review.parquet",
        )
        pq.write_table(
            pa.table(
                {
                    "product_id": [10],
                    "title": ["Lamp"],
                    "price": [19.99],
                }
            ),
            root / "product.parquet",
        )
        pq.write_table(
            pa.table({"customer_id": [1], "reviewer_name": ["Ada"]}),
            root / "customer.parquet",
        )
        self.dataset = LocalRelBenchDataset(root)
        self.graph = SchemaGraph.from_manifest(self.dataset.schema)

    def test_enumerates_direct_and_relational_attributes_up_to_hop_limit(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=2
        )
        identifiers = [candidate.id for candidate in candidates]
        self.assertIn("attribute:product:product:product.title", identifiers)
        self.assertIn("relationship:product:product>review", identifiers)
        self.assertIn("attribute:product:product>review:review.rating", identifiers)
        self.assertIn(
            "attribute:product:product>review>customer:customer.reviewer_name",
            identifiers,
        )
        customer_signal = next(
            candidate
            for candidate in candidates
            if candidate.id
            == "attribute:product:product>review>customer:customer.reviewer_name"
        )
        self.assertIn("review", customer_signal.describe(self.graph))
        self.assertIn("customer", customer_signal.describe(self.graph))
        self.assertNotIn("attribute:product:product:product.product_id", identifiers)
        self.assertNotIn(
            "attribute:product:product>review:review.product_id", identifiers
        )

    def test_hop_limit_and_source_table_bound_the_candidate_set(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source_table == "product" for candidate in candidates))
        self.assertTrue(all(candidate.hops == 0 for candidate in candidates))

    def test_decider_choice_keeps_categorical_probabilities(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )
        scorer = DeciderSignalScorer(client=_FakeDecider())
        scored = scorer.score("Find affordable products", candidates, self.graph)
        self.assertEqual(scored[0].score, 0.8)
        self.assertEqual(len(scored), len(candidates))


if __name__ == "__main__":
    unittest.main()
