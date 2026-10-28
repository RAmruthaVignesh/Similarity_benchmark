"""Offline tests for the deterministic controlled signal benchmark."""

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
from src.signals.candidates import SignalCandidate  # noqa: E402
from src.signals.scorers import ScoredSignal  # noqa: E402
from scripts.evaluate_controlled_signals import _score_task  # noqa: E402
from src.synthetic.controlled_signal_benchmark_v2 import (  # noqa: E402
    EXPECTED_CANDIDATE_COUNT,
    BENCHMARK_VERSION,
    ControlledSignalTask,
    build_controlled_signal_benchmark,
    load_tasks,
)
from src.synthetic.controlled_signal_metrics import ranking_metrics  # noqa: E402


class ControlledSignalBenchmarkTest(unittest.TestCase):
    def test_provenance_soft_metrics_collapse_attribute_path_variants_once(self):
        candidates = {
            "product-title": {
                "kind": "attribute", "attribute_table": "product",
                "attribute_column": "title", "path": ["product"],
            },
            "review-title": {
                "kind": "attribute", "attribute_table": "product",
                "attribute_column": "title", "path": ["review", "product"],
            },
            "other": {
                "kind": "attribute", "attribute_table": "product",
                "attribute_column": "price", "path": ["product"],
            },
        }
        grades = {"product-title": 3, "review-title": 0, "other": 0}
        metrics = ranking_metrics(["review-title", "product-title", "other"], grades, candidates)
        self.assertEqual(metrics["path_exact_recall@1"], 0.0)
        self.assertEqual(metrics["provenance_soft_recall@1"], 1.0)
        self.assertEqual(metrics["provenance_soft_recall@3"], 1.0)

    def test_shortlist_scoring_counts_undiscovered_qrels_as_not_retrieved(self):
        candidate = SignalCandidate(
            id="attribute:product:product:product.category",
            kind="attribute",
            source_table="product",
            path=("product",),
            attribute_table="product",
            attribute_column="category",
        )
        grades = {candidate.id: 3, "another-central-signal": 3}
        grades.update({f"distractor-{index}": 0 for index in range(37)})
        task = ControlledSignalTask(
            id="shortlist-test",
            condition="Find products by category and price.",
            source_table="product",
            reasoning_type="and-filter",
            oracle_rationale="Both category and price are central.",
            signal_grades=grades,
        )

        row = _score_task(
            "strands-decider", task, [ScoredSignal(candidate, 1.0)], graph=None,
            candidate_count=39,
        )

        self.assertEqual(row["candidate_count"], 39)
        self.assertEqual(row["returned_signal_count"], 1)
        self.assertEqual(row["recall@3"], 0.5)
        self.assertEqual(row["mrr"], 1.0)
        self.assertLess(row["ndcg@3"], 1.0)

    def _write_schema_fixture(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.yaml").write_text(
            """name: rel-amazon
manifest_version: 1
description: Amazon products, customers, and reviews.
val_timestamp: '2015-10-01'
test_timestamp: '2016-01-01'
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
""",
            encoding="utf-8",
        )
        columns = {
            "review": (
                "review_time", "customer_id", "product_id", "rating",
                "verified", "review_text", "summary",
            ),
            "product": (
                "product_id", "category", "brand", "title", "description", "price",
            ),
            "customer": ("customer_id", "customer_name"),
        }
        for table, names in columns.items():
            schema = pa.schema([(name, pa.string()) for name in names])
            pq.write_table(pa.Table.from_batches([], schema=schema), root / f"{table}.parquet")

    def test_qrels_match_full_standard_candidate_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source_root = temporary_root / "source-schema"
            root = temporary_root / "controlled"
            self._write_schema_fixture(source_root)
            built = build_controlled_signal_benchmark(root, schema_root=source_root)
            loaded = load_tasks(root)
            self.assertEqual(built, loaded)
            self.assertEqual(len(loaded), 25)
            self.assertIn(BENCHMARK_VERSION, (root / "benchmark.json").read_text())
            reasoning_types = {task.reasoning_type for task in loaded}
            self.assertGreaterEqual(len(reasoning_types), 10)
            self.assertTrue(all(task.oracle_rationale for task in loaded))

            dataset = LocalRelBenchDataset(root)
            graph = SchemaGraph.from_manifest(dataset.schema)
            candidates = enumerate_signal_candidates(dataset, graph, max_hops=2)
            self.assertEqual(len(candidates), EXPECTED_CANDIDATE_COUNT)
            source = LocalRelBenchDataset(source_root)
            source_graph = SchemaGraph.from_manifest(source.schema)
            source_candidates = enumerate_signal_candidates(
                source, source_graph, max_hops=2
            )
            self.assertEqual(
                {candidate.id for candidate in candidates},
                {candidate.id for candidate in source_candidates},
            )
            candidate_ids = {candidate.id for candidate in candidates}
            catalog = (root / "candidate_catalog.json").read_text()
            self.assertNotIn("product.color", catalog)
            self.assertNotIn("review.verified_purchase", catalog)
            for task in loaded:
                self.assertEqual(candidate_ids, set(task.signal_grades))
                self.assertIn(3, task.signal_grades.values())
                self.assertEqual(len(task.signal_grades), EXPECTED_CANDIDATE_COUNT)

    def test_customer_behavior_marks_two_hop_relationship_as_central(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source-schema"
            self._write_schema_fixture(source_root)
            tasks = build_controlled_signal_benchmark(
                root / "controlled", schema_root=source_root
            )
            task = next(task for task in tasks if task.id == "product-customer-behavior")
            self.assertEqual(task.signal_grades["relationship:product:product>review>customer"], 3)


if __name__ == "__main__":
    unittest.main()
