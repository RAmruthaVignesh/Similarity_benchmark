"""Tests for the schema graph. Run: python -m unittest discover tests"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.relbench_manifest import load_manifest, parse_manifest  # noqa: E402
from src.schema.schema_graph import Edge, SchemaGraph, SchemaGraphError  # noqa: E402

REL_AMAZON = REPO_ROOT / "data" / "manifests" / "rel-amazon" / "manifest.yaml"

REVIEW_TO_CUSTOMER = Edge("review", "customer_id", "customer", "customer_id")
REVIEW_TO_PRODUCT = Edge("review", "product_id", "product", "product_id")

# 1 prints the rel-amazon schema report before the tests run, 0 stays quiet.
VERBOSE = 1


def setUpModule() -> None:
    if VERBOSE:
        print_report(SchemaGraph.from_manifest(load_manifest(REL_AMAZON)))


def print_report(graph: SchemaGraph) -> None:
    """The graph as the schema methods see it, for eyeballing during development."""
    print("\ntables")
    for table in graph.get_tables():
        print(
            f"  {table.name:9s} pkey={table.primary_key or '-':<12s} "
            f"time_col={table.time_column or '-'}"
        )

    print("\nforeign-key edges")
    for edge in graph.get_edges():
        print(
            f"  {edge.source_table}.{edge.source_column} -> "
            f"{edge.target_table}.{edge.target_column}"
        )

    print("\nneighbors")
    for table in graph.get_tables():
        neighbors = ", ".join(graph.get_neighbors(table.name)) or "-"
        print(f"  {table.name:9s} -> {neighbors}")

    print("\npaths from product (max_hops=2)")
    for path in graph.find_paths("product", max_hops=2):
        print(f"  {' -> '.join(path)}")

    edge = graph.get_relationship("product", "review")
    print("\nget_relationship('product', 'review')")
    print(
        f"  {edge.source_table}.{edge.source_column} -> "
        f"{edge.target_table}.{edge.target_column}\n"
    )
    # unittest reports on stderr; flush so the report stays above it when piped.
    sys.stdout.flush()


def graph_from(manifest: dict) -> SchemaGraph:
    return SchemaGraph.from_manifest(parse_manifest(manifest))


class TestRelAmazonGraph(unittest.TestCase):
    """The initial development dataset: review -> customer, review -> product."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = load_manifest(REL_AMAZON)
        cls.graph = SchemaGraph.from_manifest(cls.schema)

    def test_nodes_are_the_manifest_tables(self):
        self.assertEqual(
            [table.name for table in self.graph.get_tables()],
            ["review", "product", "customer"],
        )
        for table in self.graph.get_tables():
            self.assertIs(table, self.schema.tables[table.name])

    def test_edges_resolve_the_target_primary_key(self):
        self.assertEqual(
            self.graph.get_edges(), (REVIEW_TO_CUSTOMER, REVIEW_TO_PRODUCT)
        )

    def test_neighbors_ignore_foreign_key_direction(self):
        self.assertEqual(self.graph.get_neighbors("review"), ("customer", "product"))
        self.assertEqual(self.graph.get_neighbors("product"), ("review",))
        self.assertEqual(self.graph.get_neighbors("customer"), ("review",))

    def test_relationship_is_the_same_either_way_round(self):
        self.assertEqual(
            self.graph.get_relationship("review", "product"), REVIEW_TO_PRODUCT
        )
        self.assertEqual(
            self.graph.get_relationship("product", "review"), REVIEW_TO_PRODUCT
        )

    def test_relationship_keeps_the_declared_direction(self):
        edge = self.graph.get_relationship("product", "review")
        self.assertEqual(edge.source_table, "review")
        self.assertEqual(edge.source_column, "product_id")
        self.assertEqual(edge.target_table, "product")
        self.assertEqual(edge.target_column, "product_id")

    def test_unrelated_tables_have_no_relationship(self):
        self.assertIsNone(self.graph.get_relationship("customer", "product"))

    def test_unknown_table_is_rejected(self):
        for call in (
            lambda: self.graph.get_neighbors("nope"),
            lambda: self.graph.get_relationship("review", "nope"),
            lambda: self.graph.get_relationships("nope", "review"),
            lambda: self.graph.find_paths("nope"),
        ):
            with self.assertRaises(SchemaGraphError):
                call()

    def test_one_hop_paths(self):
        self.assertEqual(
            self.graph.find_paths("product", max_hops=1), [["product", "review"]]
        )
        self.assertEqual(
            self.graph.find_paths("review", max_hops=1),
            [["review", "customer"], ["review", "product"]],
        )

    def test_two_hop_paths(self):
        self.assertEqual(
            self.graph.find_paths("product", max_hops=2),
            [["product", "review"], ["product", "review", "customer"]],
        )

    def test_paths_traverse_foreign_keys_backwards_then_forwards(self):
        # customer is only a foreign-key *target*, so leaving it at all means
        # walking review.customer_id in reverse before following review.product_id.
        self.assertEqual(
            self.graph.find_paths("customer", max_hops=2),
            [["customer", "review"], ["customer", "review", "product"]],
        )

    def test_paths_are_deterministic(self):
        self.assertEqual(
            self.graph.find_paths("product", max_hops=3),
            self.graph.find_paths("product", max_hops=3),
        )


class TestEdgeCases(unittest.TestCase):
    def test_reverse_swaps_both_endpoints(self):
        self.assertEqual(
            REVIEW_TO_PRODUCT.reverse(),
            Edge("product", "product_id", "review", "product_id"),
        )
        self.assertEqual(REVIEW_TO_PRODUCT.reverse().reverse(), REVIEW_TO_PRODUCT)

    def test_table_with_no_foreign_keys_has_no_neighbors(self):
        graph = graph_from({"name": "d", "tables": {"a": {"pkey": "id"}}})
        self.assertEqual(graph.get_edges(), ())
        self.assertEqual(graph.get_neighbors("a"), ())

    def test_self_reference_makes_a_table_its_own_neighbor(self):
        graph = graph_from(
            {
                "name": "d",
                "tables": {"staff": {"pkey": "id", "fkeys": {"manager_id": "staff"}}},
            }
        )
        self.assertEqual(graph.get_neighbors("staff"), ("staff",))
        self.assertEqual(
            graph.get_relationship("staff", "staff"),
            Edge("staff", "manager_id", "staff", "id"),
        )

    def test_neighbors_are_deduplicated(self):
        graph = graph_from(
            {
                "name": "d",
                "tables": {
                    "race": {
                        "pkey": "id",
                        "fkeys": {"home_id": "team", "away_id": "team"},
                    },
                    "team": {"pkey": "id"},
                },
            }
        )
        self.assertEqual(graph.get_neighbors("race"), ("team",))

    def test_multiple_foreign_keys_between_two_tables(self):
        graph = graph_from(
            {
                "name": "d",
                "tables": {
                    "race": {
                        "pkey": "id",
                        "fkeys": {"home_id": "team", "away_id": "team"},
                    },
                    "team": {"pkey": "id"},
                },
            }
        )
        expected = (
            Edge("race", "home_id", "team", "id"),
            Edge("race", "away_id", "team", "id"),
        )
        self.assertEqual(graph.get_relationships("team", "race"), expected)
        with self.assertRaises(SchemaGraphError):
            graph.get_relationship("team", "race")

    def test_foreign_key_to_undeclared_table_is_rejected(self):
        with self.assertRaises(SchemaGraphError):
            graph_from({"name": "d", "tables": {"a": {"fkeys": {"b_id": "b"}}}})

    def test_foreign_key_to_table_without_primary_key_is_rejected(self):
        with self.assertRaises(SchemaGraphError):
            graph_from(
                {
                    "name": "d",
                    "tables": {"a": {"fkeys": {"b_id": "b"}}, "b": {"pkey": None}},
                }
            )


class TestFindPaths(unittest.TestCase):
    CHAIN = {
        "name": "chain",
        "tables": {
            "a": {"pkey": "id", "fkeys": {"b_id": "b"}},
            "b": {"pkey": "id", "fkeys": {"c_id": "c"}},
            "c": {"pkey": "id", "fkeys": {"d_id": "d"}},
            "d": {"pkey": "id"},
        },
    }
    # A cycle: a -> b -> c -> a, so every table is reachable either way round.
    CYCLE = {
        "name": "cycle",
        "tables": {
            "a": {"pkey": "id", "fkeys": {"b_id": "b"}},
            "b": {"pkey": "id", "fkeys": {"c_id": "c"}},
            "c": {"pkey": "id", "fkeys": {"a_id": "a"}},
        },
    }

    def test_max_hops_is_respected(self):
        graph = graph_from(self.CHAIN)
        self.assertEqual(graph.find_paths("a", max_hops=0), [])
        self.assertEqual(graph.find_paths("a", max_hops=1), [["a", "b"]])
        self.assertEqual(
            graph.find_paths("a", max_hops=2), [["a", "b"], ["a", "b", "c"]]
        )
        self.assertEqual(
            graph.find_paths("a", max_hops=3),
            [["a", "b"], ["a", "b", "c"], ["a", "b", "c", "d"]],
        )

    def test_max_hops_defaults_to_two(self):
        graph = graph_from(self.CHAIN)
        self.assertEqual(graph.find_paths("a"), graph.find_paths("a", max_hops=2))

    def test_no_table_repeats_within_a_path(self):
        graph = graph_from(self.CYCLE)
        paths = graph.find_paths("a", max_hops=5)
        for path in paths:
            self.assertEqual(len(path), len(set(path)), f"repeat in {path}")
        # Three tables in a cycle: at most two hops away before running out.
        self.assertEqual(
            paths,
            [["a", "b"], ["a", "b", "c"], ["a", "c"], ["a", "c", "b"]],
        )

    def test_a_cycle_is_walked_in_both_directions(self):
        graph = graph_from(self.CYCLE)
        # a -> b follows a.b_id forwards; a -> c follows c.a_id backwards.
        self.assertEqual(graph.find_paths("a", max_hops=1), [["a", "b"], ["a", "c"]])

    def test_table_with_no_foreign_keys_has_no_paths(self):
        graph = graph_from({"name": "d", "tables": {"a": {"pkey": "id"}}})
        self.assertEqual(graph.find_paths("a", max_hops=3), [])

    def test_self_reference_yields_no_path(self):
        graph = graph_from(
            {
                "name": "d",
                "tables": {"staff": {"pkey": "id", "fkeys": {"manager_id": "staff"}}},
            }
        )
        self.assertEqual(graph.find_paths("staff", max_hops=3), [])

    def test_duplicate_foreign_keys_yield_one_path(self):
        graph = graph_from(
            {
                "name": "d",
                "tables": {
                    "race": {
                        "pkey": "id",
                        "fkeys": {"home_id": "team", "away_id": "team"},
                    },
                    "team": {"pkey": "id"},
                },
            }
        )
        self.assertEqual(graph.find_paths("race", max_hops=2), [["race", "team"]])


if __name__ == "__main__":
    unittest.main()
