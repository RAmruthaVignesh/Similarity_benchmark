"""Enumerate inspectable schema signals from relational tables and paths.

The first signal-discovery baseline intentionally uses only information visible
in the local relational schema: ordinary columns and the foreign-key paths
through which their values can be reached.  It does not inspect data values or
try to infer relevance while generating candidates.  A later scorer receives
the same fixed candidate list for every condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.datasets.relational_dataset import RelationalDataset
from src.schema.schema_graph import Edge, SchemaGraph

__all__ = [
    "SignalCandidate",
    "enumerate_signal_candidates",
]


@dataclass(frozen=True)
class SignalCandidate:
    """One schema-level attribute or relationship available to an anchor table."""

    id: str
    kind: str
    source_table: str
    path: tuple[str, ...]
    attribute_table: str | None = None
    attribute_column: str | None = None

    @property
    def hops(self) -> int:
        """Number of foreign-key joins needed to reach this candidate."""
        return len(self.path) - 1

    def describe(self, graph: SchemaGraph) -> str:
        """Return a stable natural-language description for a decision model."""
        path = _describe_path(self.path, graph)
        if self.kind == "relationship":
            return f"Relationship path: {path}."
        assert self.attribute_table is not None
        assert self.attribute_column is not None
        return f"Attribute {self.attribute_table}.{self.attribute_column}; path: {path}."

    def to_dict(self, graph: SchemaGraph) -> dict[str, object]:
        """A JSON-friendly representation, including the model-facing text."""
        return {
            "id": self.id,
            "kind": self.kind,
            "source_table": self.source_table,
            "path": list(self.path),
            "hops": self.hops,
            "attribute_table": self.attribute_table,
            "attribute_column": self.attribute_column,
            "description": self.describe(graph),
        }


def enumerate_signal_candidates(
    dataset: RelationalDataset[object],
    graph: SchemaGraph,
    *,
    source_tables: Sequence[str] | None = None,
    max_hops: int = 2,
    include_relationships: bool = True,
    include_structural_columns: bool = False,
) -> tuple[SignalCandidate, ...]:
    """Return deterministic, schema-only signal candidates.

    Each selected source table contributes direct attributes and attributes of
    tables reachable by at most ``max_hops`` foreign-key joins.  A relational
    attribute remains distinct for each source/path pair: ``product -> review
    -> customer`` is a different usable signal from ``customer -> review ->
    product``.  Paths do not repeat tables, following :meth:`SchemaGraph.find_paths`.

    Identifiers and declared foreign-key columns are excluded by default. They
    describe database structure rather than a meaningful user-facing signal;
    relationship candidates preserve that structure explicitly.  Time columns
    remain eligible because recency and temporal behaviour may be relevant.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")

    available_tables = dataset.get_tables()
    graph_tables = tuple(table.name for table in graph.get_tables())
    if tuple(available_tables) != graph_tables:
        raise ValueError(
            "dataset tables do not match the schema graph: "
            f"dataset={tuple(available_tables)!r}, graph={graph_tables!r}"
        )

    roots = tuple(source_tables or available_tables)
    unknown = [table for table in roots if table not in available_tables]
    if unknown:
        raise ValueError(f"unknown source table(s): {', '.join(unknown)}")

    candidates: list[SignalCandidate] = []
    seen: set[str] = set()
    for source_table in roots:
        paths: Iterable[list[str]] = ([source_table], *graph.find_paths(source_table, max_hops))
        for raw_path in paths:
            path = tuple(raw_path)
            if len(path) > 1 and include_relationships:
                relationship = SignalCandidate(
                    id=_candidate_id("relationship", source_table, path),
                    kind="relationship",
                    source_table=source_table,
                    path=path,
                )
                _append_once(candidates, seen, relationship)

            target_table = path[-1]
            for column in dataset.get_column_names(target_table):
                if not include_structural_columns and _is_structural_column(
                    dataset, target_table, column
                ):
                    continue
                attribute = SignalCandidate(
                    id=_candidate_id(
                        "attribute", source_table, path, target_table, column
                    ),
                    kind="attribute",
                    source_table=source_table,
                    path=path,
                    attribute_table=target_table,
                    attribute_column=column,
                )
                _append_once(candidates, seen, attribute)
    return tuple(candidates)


def _append_once(
    candidates: list[SignalCandidate], seen: set[str], candidate: SignalCandidate
) -> None:
    if candidate.id not in seen:
        candidates.append(candidate)
        seen.add(candidate.id)


def _candidate_id(
    kind: str,
    source_table: str,
    path: tuple[str, ...],
    attribute_table: str | None = None,
    attribute_column: str | None = None,
) -> str:
    path_part = ">".join(path)
    suffix = (
        ""
        if attribute_table is None
        else f":{attribute_table}.{attribute_column}"
    )
    return f"{kind}:{source_table}:{path_part}{suffix}"


def _is_structural_column(
    dataset: RelationalDataset[object], table: str, column: str
) -> bool:
    if column == dataset.get_primary_key(table):
        return True
    return any(
        source == table and foreign_key.column == column
        for source, foreign_key in dataset.get_foreign_keys()
    )


def _describe_path(path: tuple[str, ...], graph: SchemaGraph) -> str:
    if len(path) == 1:
        return path[0]
    parts = [path[0]]
    for source, target in zip(path, path[1:]):
        edge = _edge_for_step(graph, source, target)
        if edge.source_table == source:
            parts.append(
                f"--{edge.source_table}.{edge.source_column}→"
                f"{edge.target_table}.{edge.target_column}-- {target}"
            )
        else:
            parts.append(
                f"--{edge.target_table}.{edge.target_column}←"
                f"{edge.source_table}.{edge.source_column}-- {target}"
            )
    return " ".join(parts)


def _edge_for_step(graph: SchemaGraph, source: str, target: str) -> Edge:
    edges = graph.get_relationships(source, target)
    if len(edges) != 1:
        raise ValueError(
            f"path step {source!r} -> {target!r} is ambiguous; "
            "signal enumeration requires exactly one foreign key per table pair"
        )
    return edges[0]
