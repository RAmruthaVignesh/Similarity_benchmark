"""Offline tests for LLM-judge prompt, parsing, cache, and metric semantics."""

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
from src.evaluation.ranking_metrics import ranking_metrics  # noqa: E402
from src.evaluation.llm_judge import (  # noqa: E402
    JudgeConfig,
    build_judge_prompt,
    cache_key,
    describe_schema,
    parse_judge_response,
    schema_version,
)
from src.schema.schema_graph import SchemaGraph  # noqa: E402
from src.signals.candidates import enumerate_signal_candidates  # noqa: E402


class LLMJudgeTest(unittest.TestCase):
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
      product_id: product
  product:
    pkey: product_id
    time_col: null
    fkeys: {}
""".lstrip()
        )
        pq.write_table(pa.table({"review_time": ["2020-01-01"], "product_id": [1], "rating": [5]}), root / "review.parquet")
        pq.write_table(pa.table({"product_id": [1], "title": ["Lamp"], "price": [10.0]}), root / "product.parquet")
        dataset = LocalRelBenchDataset(root)
        self.graph = SchemaGraph.from_manifest(dataset.schema)
        self.candidates = enumerate_signal_candidates(dataset, self.graph, max_hops=1)

    def test_prompt_is_blind_to_method_rank_and_score(self):
        schema = describe_schema(self.candidates, self.graph)
        prompt = build_judge_prompt("Find affordable products", self.candidates[0], self.graph, schema)
        rendered = "\n".join(message["content"] for message in prompt).lower()
        self.assertIn("find affordable products", rendered)
        self.assertIn("tables and attributes", rendered)
        self.assertNotIn("minilm", rendered)
        self.assertNotIn("decider", rendered)

    def test_response_parsing_is_strict(self):
        self.assertEqual(parse_judge_response('{"status":"judged","grade":3,"reason":"Direct price evidence."}')["grade"], 3)
        self.assertEqual(parse_judge_response('{"status":"unable_to_judge","grade":null,"reason":"Ambiguous intent."}')["status"], "unable_to_judge")
        self.assertEqual(parse_judge_response('{"status":"judged","grade":4,"reason":"bad"}')["status"], "invalid_response")
        self.assertEqual(parse_judge_response("not json")["status"], "invalid_response")

    def test_cache_key_changes_when_semantic_inputs_change(self):
        candidate = self.candidates[0]
        version = schema_version(self.candidates, self.graph, {"hop_limit": 1})
        config = JudgeConfig("mistral", "example/model", "commit-a")
        first = cache_key(condition_id="one", condition="Find lamps", candidate=candidate, schema_version=version, config=config)
        second = cache_key(condition_id="one", condition="Find lamps", candidate=candidate, schema_version=version, config=JudgeConfig("mistral", "example/model", "commit-b"))
        self.assertNotEqual(first, second)

    def test_metrics_do_not_count_missing_judgments_as_relevant_or_irrelevant(self):
        metrics = ranking_metrics(["known-relevant", "unjudged", "known-irrelevant"], {"known-relevant": 3, "known-irrelevant": 0}, 2)
        self.assertEqual(metrics["ndcg@1"], 1.0)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["precision@3"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
