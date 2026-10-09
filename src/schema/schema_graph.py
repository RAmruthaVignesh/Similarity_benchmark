"""A graph view of a parsed RelBench manifest: tables as nodes, foreign keys as edges.

This wraps :class:`~src.datasets.relbench_manifest.DatasetSchema` rather than
restating it -- nodes *are* the manifest's ``TableSchema`` objects. The one thing
the graph adds is the join's other endpoint: a manifest foreign key names only
its target table, so the target column is resolved here from that table's
primary key. Each edge is stored once, in the direction the foreign key points,
and carries both endpoints so it can be traversed either way: lookups by table
are direction-agnostic, while the edges they return keep the declared direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.datasets.relbench_manifest import DatasetSchema, TableSchema

__all__ = [
    "Edge",
    "SchemaGraph",
    "SchemaGraphError",
]


class SchemaGraphError(ValueError):
    """Raised when a manifest's foreign keys do not describe a joinable graph."""


@dataclass(frozen=True)
class Edge:
    """One foreign key as a directed join: source column -> target column."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str

    def reverse(self) -> Edge:
        """The same relationship, traversed from the target table back to the source."""
        return Edge(
            source_table=self.target_table,
            source_column=self.target_column,
            target_table=self.source_table,
            target_column=self.source_column,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
        }


@dataclass(frozen=True)
class SchemaGraph:
    """The foreign-key graph of a dataset, built from its parsed manifest."""

    dataset: DatasetSchema
    edges: tuple[Edge, ...] = ()

    @classmethod
    def from_manifest(cls, manifest: DatasetSchema) -> SchemaGraph:
        """Build the graph for a parsed manifest.

        Raises:
            SchemaGraphError: if a foreign key targets a table the manifest does
                not declare, or one that declares no primary key to join on.
        """
        edges = []
        for table in manifest.tables.values():
            for fkey in table.foreign_keys:
                target = manifest.tables.get(fkey.target_table)
                if target is None:
                    raise SchemaGraphError(
                        f"table '{table.name}': foreign key '{fkey.column}' targets "
                        f"undeclared table '{fkey.target_table}'"
                    )
                if target.primary_key is None:
                    raise SchemaGraphError(
                        f"table '{table.name}': foreign key '{fkey.column}' targets "
                        f"table '{target.name}', which declares no primary key to "
                        f"join on"
                    )
                edges.append(
                    Edge(
                        source_table=table.name,
                        source_column=fkey.column,
                        target_table=target.name,
                        target_column=fkey.target_column or target.primary_key,
                    )
                )
        return cls(dataset=manifest, edges=tuple(edges))

    def get_tables(self) -> tuple[TableSchema, ...]:
        """The graph's nodes, in manifest order."""
        return tuple(self.dataset.tables.values())

    def get_edges(self) -> tuple[Edge, ...]:
        """Every foreign key, each in the direction it points."""
        return self.edges

    def get_neighbors(self, table_name: str) -> tuple[str, ...]:
        """Tables directly joinable to ``table_name``, whichever way the keys point.

        A table with a foreign key onto itself is its own neighbor.
        """
        self._require_table(table_name)
        neighbors = []
        for edge in self.edges:
            if edge.source_table == table_name:
                neighbors.append(edge.target_table)
            if edge.target_table == table_name:
                neighbors.append(edge.source_table)
        return tuple(dict.fromkeys(neighbors))

    def get_relationships(self, table_a: str, table_b: str) -> tuple[Edge, ...]:
        """Every foreign key joining the two tables, each in the direction it points.

        The pair is unordered: the edges come back as the manifest declares them,
        so swapping the arguments returns the same result.
        """
        self._require_table(table_a)
        self._require_table(table_b)
        pair = {table_a, table_b}
        return tuple(
            edge
            for edge in self.edges
            if {edge.source_table, edge.target_table} == pair
        )

    def get_relationship(self, table_a: str, table_b: str) -> Edge | None:
        """The foreign key joining the two tables, or ``None`` if they are unrelated.

        Raises:
            SchemaGraphError: if more than one foreign key joins the pair; use
                :meth:`get_relationships` when that is expected.
        """
        edges = self.get_relationships(table_a, table_b)
        if not edges:
            return None
        if len(edges) > 1:
            joins = ", ".join(
                f"{e.source_table}.{e.source_column} -> "
                f"{e.target_table}.{e.target_column}"
                for e in edges
            )
            raise SchemaGraphError(
                f"tables '{table_a}' and '{table_b}' are joined by "
                f"{len(edges)} foreign keys ({joins}); use get_relationships()"
            )
        return edges[0]

    def find_paths(self, source_table: str, max_hops: int = 2) -> list[list[str]]:
        """Simple paths of 1 to ``max_hops`` hops leaving ``source_table``.

        Foreign keys are followed in either direction and no table repeats within
        a path, so a path ends once its neighbors are all already on it. Each
        path is a list of table names starting with ``source_table``.

        The walk is depth-first over :meth:`get_neighbors`, so the order is
        deterministic: a path is followed immediately by its own extensions, and
        neighbors come in the order the manifest declares their foreign keys.
        """
        self._require_table(source_table)
        paths: list[list[str]] = []
        self._extend([source_table], max_hops, paths)
        return paths

    def _extend(
        self, path: list[str], hops_left: int, paths: list[list[str]]
    ) -> None:
        if hops_left <= 0:
            return
        for neighbor in self.get_neighbors(path[-1]):
            if neighbor in path:
                continue
            extended = [*path, neighbor]
            paths.append(extended)
            self._extend(extended, hops_left - 1, paths)

    def _require_table(self, table_name: str) -> None:
        if table_name not in self.dataset.tables:
            raise SchemaGraphError(
                f"unknown table '{table_name}'; "
                f"manifest '{self.dataset.name}' declares "
                f"{', '.join(self.dataset.table_names) or 'none'}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.name,
            "tables": [table.name for table in self.get_tables()],
            "edges": [edge.to_dict() for edge in self.get_edges()],
        }
