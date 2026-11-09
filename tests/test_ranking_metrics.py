"""Checks for human-qrel ranking metrics."""

import unittest

from src.evaluation.ranking_metrics import ranking_metrics


class RankingMetricsTest(unittest.TestCase):
    def test_unjudged_candidates_are_not_counted_as_relevant(self):
        metrics = ranking_metrics(
            ["known-relevant", "unjudged", "known-irrelevant"],
            {"known-relevant": 3, "known-irrelevant": 0},
            2,
        )
        self.assertEqual(metrics["ndcg@1"], 1.0)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["precision@3"], 1 / 3)
