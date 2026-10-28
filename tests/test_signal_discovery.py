"""Tests for deterministic schema candidates and local scoring adapters."""

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
from src.signals.candidates import candidate_from_dict, enumerate_signal_candidates  # noqa: E402
from src.signals.graph_navigation import (  # noqa: E402
    DeciderGraphNavigator,
    LLMGraphNavigator,
    format_search_history,
)
from src.signals.llm_discovery import (  # noqa: E402
    LLMDiscoveryConfig,
    LLMPointwiseSignalScorer,
    build_action_prompt,
    build_signal_prompt,
    parse_utility_response,
)
from src.signals.scorers import (  # noqa: E402
    DeciderSignalScorer,
    RandomSignalScorer,
    _ranking_logits,
    _format_intent_context,
)


class _FakeDecider:
    def decide(self, state, questions):
        self.state = state
        self.questions = questions
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


class _NavigationFakeDecider:
    """Deterministic choices that exercise root selection, a join, and emission."""

    def __init__(self):
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        answers = []
        for question in questions:
            options = question["options"]
            if "starting point" in question["question"]:
                selected = next(option for option in options if option.endswith("product"))
            elif "Current path: product\n" in state:
                selected = next(
                    option for option in options if option.startswith("Traverse relationship:")
                )
            elif "Signals already selected across explored branches: Attribute review.rating" in state:
                selected = next(option for option in options if option.startswith("Stop"))
            elif "Current path: product --product.product_id" in state:
                selected = next(
                    option for option in options if "Select attribute: review.rating" in option
                )
            else:
                selected = next(option for option in options if option.startswith("Stop"))
            answers.append(
                {
                    "choice": selected,
                    "probs": {
                        option: 0.9 if option == selected else 0.1 / (len(options) - 1)
                        for option in options
                    },
                }
            )
        return answers


class _FakePointwiseLLM:
    """Offline action scorer that ranks the first legal option highest."""

    model_name = "fake-pointwise-llm"

    def score_actions(self, context, question, actions, action_kinds):
        del context, question, action_kinds
        return [
            {"status": "scored", "utility_score": 100 - index, "evidence": f"action {index}", "attempts": 1}
            for index, _ in enumerate(actions)
        ]


class _FakeSignalLLM(LLMPointwiseSignalScorer):
    def _generate(self, messages):
        # Deliberately ties every candidate to exercise deterministic ID ordering.
        return ['{"utility_score": 50, "evidence": "schema evidence"}'] * len(messages)


class _ZeroTiePointwiseLLM:
    """Offline scorer that gives every legal action the same utility."""

    model_name = "zero-tie-pointwise-llm"

    def score_actions(self, context, question, actions, action_kinds):
        del context, question, action_kinds
        return [
            {"status": "scored", "utility_score": 0, "evidence": "no progress", "attempts": 1}
            for _ in actions
        ]


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

    def test_pointwise_utility_parser_rejects_invalid_values(self):
        self.assertEqual(
            parse_utility_response('{"utility_score": 50, "evidence": "schema grounded"}')["status"],
            "scored",
        )
        self.assertEqual(
            parse_utility_response('{"utility_score": 101, "evidence": "bad"}')["status"],
            "invalid_response",
        )

    def test_pointwise_utility_parser_recovers_valid_score_from_bad_evidence_quoting(self):
        parsed = parse_utility_response(
            '{"utility_score": 62, "evidence": "Product text is "useful" context"}'
        )
        self.assertEqual(parsed["status"], "scored")
        self.assertEqual(parsed["utility_score"], 62)
        self.assertEqual(parsed["parse_status"], "recovered_score")

    def test_llm_graph_navigator_retains_ordinal_action_trace(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=1)
        result = LLMGraphNavigator(
            _FakePointwiseLLM(), max_hops=1, max_decisions=2, branch_width=1
        ).discover("Find useful products", candidates, self.graph)
        self.assertTrue(result.trace)
        scored_actions = result.trace[0]["llm_pointwise_actions"]
        self.assertEqual(scored_actions[0]["ordinal_rank"], 1)
        self.assertEqual(result.trace[0]["frontier_selection"], "within_menu_ordinal_rank")

    def test_llm_graph_ties_prefer_exploration_to_stopping(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=1)
        result = LLMGraphNavigator(
            _ZeroTiePointwiseLLM(), max_hops=1, max_decisions=1, branch_width=1
        ).discover("Find useful products", candidates, self.graph)
        root = result.trace[0]
        self.assertTrue(root["model_choice"].startswith("Start from table:"))
        self.assertTrue(all(action["kind"] == "start" for action in root["legal_actions"]))

    def test_graph_action_prompts_explain_control_actions(self):
        root = "\n".join(message["content"] for message in build_action_prompt(
            "state", "question", "Start from table: product", "start"
        ))
        traverse = "\n".join(message["content"] for message in build_action_prompt(
            "state", "question", "Traverse relationship: product to review", "traverse"
        ))
        select = "\n".join(message["content"] for message in build_action_prompt(
            "state", "question", "Select attribute: product.price", "select_attribute"
        ))
        self.assertIn("root-selection action, not a signal", root)
        self.assertIn("traversal action, not a signal", traverse)
        self.assertIn("ordinary usefulness", select)

    def test_llm_pointwise_scorer_is_tie_stable_and_prompt_has_no_grades(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=0)
        scorer = _FakeSignalLLM(LLMDiscoveryConfig(
            model_family="mistral", model_name="fake", batch_size=4
        ))
        ranked = scorer.score("Find products", candidates, self.graph)
        self.assertEqual([item.candidate.id for item in ranked], sorted(candidate.id for candidate in candidates))
        self.assertTrue(all(item.metadata and item.metadata["utility_score"] == 50 for item in ranked))
        prompt = build_signal_prompt("Find products", candidates[0], self.graph, "Schema: x")
        self.assertNotIn("Relevance grades", "\n".join(message["content"] for message in prompt))

    def test_human_signal_explanation_and_diagram_preserve_the_path(self):
        candidate = next(
            value
            for value in enumerate_signal_candidates(
                self.dataset, self.graph, source_tables=["customer"], max_hops=2
            )
            if value.id == "attribute:customer:customer>review>product:product.title"
        )

        self.assertEqual(
            candidate.explain_for_human(self.graph),
            "Start from a customer, follow their reviews to the products they reviewed, "
            "and use those products' titles as the signal.",
        )
        self.assertEqual(
            candidate.human_diagram(self.graph),
            "customer\n"
            "   └─ reviews, linked by customer_id\n"
            "      └─ products, linked by product_id\n"
            "         └─ product.title",
        )
        restored = candidate_from_dict(candidate.to_dict(self.graph))
        self.assertEqual(restored, candidate)

    def test_hop_limit_and_source_table_bound_the_candidate_set(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )
        self.assertTrue(candidates)
        self.assertTrue(all(candidate.source_table == "product" for candidate in candidates))
        self.assertTrue(all(candidate.hops == 0 for candidate in candidates))

    def test_feature_collapsed_catalog_keeps_one_direct_attribute_per_column(self):
        candidates = enumerate_signal_candidates(
            self.dataset,
            self.graph,
            max_hops=2,
            catalog_view="feature-collapsed",
        )
        identifiers = [candidate.id for candidate in candidates]
        self.assertIn("attribute:product:product:product.title", identifiers)
        self.assertIn("attribute:review:review:review.rating", identifiers)
        self.assertIn("attribute:customer:customer:customer.reviewer_name", identifiers)
        self.assertTrue(all(candidate.kind == "attribute" for candidate in candidates))
        self.assertTrue(all(candidate.hops == 0 for candidate in candidates))
        self.assertTrue(
            all(candidate.source_table == candidate.attribute_table for candidate in candidates)
        )
        self.assertNotIn("attribute:product:product>review:review.rating", identifiers)
        self.assertNotIn("relationship:product:product>review", identifiers)

    def test_get_column_names_reads_parquet_schema_without_loading_table(self):
        with patch.object(
            self.dataset,
            "get_table",
            side_effect=AssertionError("column lookup must not load table rows"),
        ):
            self.assertEqual(
                self.dataset.get_column_names("product"), ("product_id", "title", "price")
            )

    def test_decider_choice_keeps_categorical_probabilities(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )
        scorer = DeciderSignalScorer(client=_FakeDecider())
        scored = scorer.score("Find affordable products", candidates, self.graph)
        self.assertEqual(scored[0].score, 0.8)
        self.assertEqual(len(scored), len(candidates))

    def test_random_baseline_is_reproducible_and_seeded(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=1
        )
        first = RandomSignalScorer(seed=17).score("Find useful products", candidates, self.graph)
        repeated = RandomSignalScorer(seed=17).score("Find useful products", candidates, self.graph)
        other_seed = RandomSignalScorer(seed=18).score("Find useful products", candidates, self.graph)

        self.assertEqual(
            [item.candidate.id for item in first],
            [item.candidate.id for item in repeated],
        )
        self.assertNotEqual(
            [item.candidate.id for item in first],
            [item.candidate.id for item in other_seed],
        )

    def test_decider_graph_reranking_uses_search_history_not_full_schema(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )[:2]
        client = _FakeDecider()
        scorer = DeciderSignalScorer(client=client)
        ranked = scorer.score(
            "Find affordable products",
            candidates,
            self.graph,
            history_context=(
                "Decision 1 (starting-table selection): Decider chose: Start from table: product\n"
                "Signals collected:\n- Attribute product.price; path: product"
            ),
        )

        self.assertEqual({row.candidate.id for row in ranked}, {row.id for row in candidates})
        self.assertEqual(ranked[0].candidate, candidates[0])
        self.assertAlmostEqual(sum(row.score for row in ranked), 1.0)
        self.assertEqual(len(client.questions[0]["options"]), len(candidates))
        self.assertIn("Graph-search history:", client.state)
        self.assertIn("Start from table: product", client.state)
        self.assertNotIn("Tables and attributes:", client.state)
        self.assertNotIn("review: body, rating, review_time", client.state)

    def test_decider_reranking_singleton_shortlist_needs_no_model_choice(self):
        candidate = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=0
        )[0]
        ranked = DeciderSignalScorer().score(
            "Find affordable products", [candidate], self.graph
        )
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].score, 1.0)

    def test_decider_prompt_includes_schema_and_readable_paths_without_ids(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, source_tables=["product"], max_hops=2
        )
        context = _format_intent_context(
            "Find products reviewed by similar customers", candidates, self.graph
        )
        client = _FakeDecider()
        DeciderSignalScorer(client=client).score(
            "Find products reviewed by similar customers", candidates, self.graph
        )

        self.assertEqual(client.state, context)
        self.assertIn("Tables and attributes:", client.state)
        self.assertIn("product: price, title", client.state)
        self.assertIn("review.customer_id references customer.customer_id", client.state)
        self.assertIn("User intent: Find products reviewed by similar customers", client.state)
        options = client.questions[0]["options"]
        self.assertTrue(any("product.title; path: product" in option for option in options))
        self.assertTrue(
            any(
                "review.rating; path: product" in option
                and "product.product_id←review.product_id" in option
                for option in options
            )
        )
        self.assertTrue(all(not option.startswith("attribute:") for option in options))

    def test_decider_graph_navigation_has_legal_actions_and_stateful_trace(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, max_hops=2
        )
        client = _NavigationFakeDecider()
        result = DeciderGraphNavigator(
            client=client,
            max_hops=2,
            max_decisions=4,
            branch_width=1,
            max_signals=2,
        ).discover("Find products with strong ratings", candidates, self.graph)

        self.assertEqual(result.model_calls, 4)
        self.assertIn(
            "attribute:product:product>review:review.rating",
            {candidate.id for candidate in result.signals},
        )
        self.assertEqual(result.trace[0]["current_path"], None)
        self.assertEqual(result.trace[0]["model_choice"], "Start from table: product")
        self.assertNotIn(
            "Finish discovery without selecting a start table.",
            [action["option"] for action in result.trace[0]["legal_actions"]],
        )
        self.assertEqual(result.trace[1]["current_path"], ["product"])
        self.assertIn("Current path: product", result.trace[1]["state_context"])
        self.assertIn(
            "Prior decisions on this branch: Start from table: product",
            result.trace[1]["state_context"],
        )
        for decision in result.trace:
            options = [action["option"] for action in decision["legal_actions"]]
            self.assertEqual(len(options), len(set(options)))
            for action in decision["expanded_actions"]:
                self.assertIn(action["option"], options)
        for decision in result.trace[1:]:
            path = decision["current_path"]
            assert path is not None
            self.assertEqual(len(path), len(set(path)))

    def test_graph_search_cannot_finish_before_selecting_a_signal(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=2)
        result = DeciderGraphNavigator(
            client=_NavigationFakeDecider(),
            max_hops=2,
            max_decisions=2,
            branch_width=1,
            max_signals=2,
        ).discover("Find products with strong ratings", candidates, self.graph)
        for event in result.trace:
            if not event["current_path"]:
                continue
            self.assertNotIn(
                "Finish discovery and return the signals collected so far.",
                [action["option"] for action in event["legal_actions"]],
            )

    def test_graph_search_closes_a_one_option_leaf_without_calling_the_model(self):
        from src.signals.graph_navigation import _Action

        class _LeafNavigator(DeciderGraphNavigator):
            def _legal_actions(self, state, catalog, graph, discovered):
                del state, catalog, graph, discovered
                return (_Action("stop_branch", "Stop this path; continue any other queued paths."),)

        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=0)
        result = _LeafNavigator(
            client=_NavigationFakeDecider(),
            max_hops=0,
            max_decisions=3,
            branch_width=1,
            max_signals=2,
        ).discover("Find products", candidates, self.graph)
        # One root-table decision is made; the leaf has only stop_branch and
        # is closed by the program rather than sent to a two-choice Decider.
        self.assertEqual(result.model_calls, 1)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.signals, ())

    def test_model_selected_root_branches_to_top_tables(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=2)
        client = _NavigationFakeDecider()
        result = DeciderGraphNavigator(
            client=client,
            max_hops=2,
            max_decisions=1,
            branch_width=3,
            max_signals=2,
        ).discover("Find products with strong ratings", candidates, self.graph)

        root_event = result.trace[0]
        self.assertEqual(root_event["model_choice"], "Start from table: product")
        root_tables = [
            action["target_table"]
            for action in root_event["expanded_actions"]
            if action["kind"] == "start"
        ]
        self.assertEqual(len(root_tables), 3)
        self.assertEqual(root_event["expanded_actions"][0]["target_table"], "product")
        self.assertEqual(result.model_calls, 1)

    def test_final_prompt_history_summarizes_graph_trace_without_schema_context(self):
        candidates = enumerate_signal_candidates(
            self.dataset, self.graph, max_hops=2
        )
        client = _NavigationFakeDecider()
        result = DeciderGraphNavigator(
            client=client,
            max_hops=2,
            max_decisions=1,
            branch_width=1,
            max_signals=2,
        ).discover(
            "Find products with strong ratings",
            candidates,
            self.graph,
        )
        history = format_search_history(result, self.graph)
        self.assertIn("Decision 1 (starting-table selection)", history)
        self.assertIn("Decider chose:", history)
        self.assertNotIn("Tables and attributes:", history)
        self.assertNotIn("state_context", history)

    def test_oracle_root_forces_start_table_and_reserves_equal_search_budget(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=2)
        client = _NavigationFakeDecider()
        result = DeciderGraphNavigator(
            client=client,
            max_hops=2,
            max_decisions=4,
            branch_width=1,
            max_signals=2,
        ).discover(
            "Find products with strong ratings",
            candidates,
            self.graph,
            start_table="product",
        )

        self.assertEqual(result.model_calls, 3)
        self.assertEqual(len(client.calls), 3)
        self.assertTrue(all("starting point" not in call[1][0]["question"] for call in client.calls))
        self.assertEqual(result.trace[0]["current_path"], ["product"])
        history = format_search_history(result, self.graph, forced_root="product")
        self.assertIn("Oracle-forced starting table: product", history)
        self.assertNotIn("Tables and attributes:", history)

    def test_oracle_root_rejects_unknown_table(self):
        candidates = enumerate_signal_candidates(self.dataset, self.graph, max_hops=2)
        with self.assertRaisesRegex(ValueError, "unknown graph-search start table"):
            DeciderGraphNavigator(
                client=_NavigationFakeDecider(), max_decisions=3
            ).discover("Find products", candidates, self.graph, start_table="missing")

    def test_ranking_logits_preserve_raw_scale_for_softmax(self):
        one_logit = np.array([[-2.0], [0.5]])
        two_logits = np.array([[1.0, 3.0], [4.0, 2.0]])

        np.testing.assert_allclose(_ranking_logits(one_logit), [-2.0, 0.5])
        np.testing.assert_allclose(_ranking_logits(two_logits), [2.0, -2.0])


if __name__ == "__main__":
    unittest.main()
