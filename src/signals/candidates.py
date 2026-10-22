"""Enumerate inspectable schema signals from relational tables and paths.

The first signal-discovery baseline intentionally uses only information visible
in the local relational schema: ordinary columns and the foreign-key paths
through which their values can be reached.  It does not inspect data values or
try to infer relevance while generating candidates.  A later scorer receives
the same fixed candidate list for every condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from src.datasets.relational_dataset import RelationalDataset
from src.schema.schema_graph import Edge, SchemaGraph

__all__ = [
    "SignalCandidate",
    "CATALOG_VIEWS",
    "candidate_from_dict",
    "enumerate_signal_candidates",
]

CATALOG_VIEWS = ("path-aware", "feature-collapsed")
CatalogView = Literal["path-aware", "feature-collapsed"]


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

    def explain_for_human(self, graph: SchemaGraph) -> str:
        """Explain the path-specific signal in plain language for annotation UIs."""
        source = _human_table_name(self.path[0], plural=False)
        target = _human_table_name(self.path[-1], plural=False)
        path_phrase = _human_path_phrase(self.path, graph)
        if self.kind == "relationship":
            if self.hops == 1:
                return (
                    f"Start from a {source}, follow {path_phrase}; "
                    "the relationship itself is the signal."
                )
            return (
                f"Start from a {source}, follow {path_phrase}; "
                "the relationship path itself is the signal."
            )

        assert self.attribute_table is not None
        assert self.attribute_column is not None
        attribute = _human_attribute_name(self.attribute_column, plural=self.hops > 0)
        if self.hops == 0:
            return f"Start from a {source} and use its {attribute} as the signal."
        target_plural = _human_table_name(target, plural=True)
        return (
            f"Start from a {source}, follow {path_phrase}, and use those "
            f"{target_plural}' {attribute} as the signal."
        )

    def human_diagram(self, graph: SchemaGraph) -> str:
        """Render a compact, readable graph walk for annotation and diagnostics."""
        lines = [self.path[0]]
        indent = ""
        for source, target in zip(self.path, self.path[1:]):
            edge = _edge_for_step(graph, source, target)
            foreign_key = edge.source_column
            indent += "   "
            lines.append(
                f"{indent}└─ {_human_table_name(target, plural=True)}, linked by {foreign_key}"
            )
        if self.kind == "relationship":
            indent += "   "
            lines.append(f"{indent}└─ relationship path")
        else:
            assert self.attribute_table is not None
            assert self.attribute_column is not None
            indent += "   "
            lines.append(f"{indent}└─ {self.attribute_table}.{self.attribute_column}")
        return "\n".join(lines)

    def to_dict(self, graph: SchemaGraph) -> dict[str, object]:
        """A JSON-friendly representation, including model and human-facing text."""
        return {
            "id": self.id,
            "kind": self.kind,
            "source_table": self.source_table,
            "path": list(self.path),
            "hops": self.hops,
            "attribute_table": self.attribute_table,
            "attribute_column": self.attribute_column,
            "description": self.describe(graph),
            "human_explanation": self.explain_for_human(graph),
            "path_diagram": self.human_diagram(graph),
        }


def candidate_from_dict(data: dict[str, object]) -> SignalCandidate:
    """Recreate a candidate stored in a result or annotation JSONL row."""
    path = data.get("path")
    if not isinstance(path, list) or not all(isinstance(table, str) for table in path):
        raise ValueError("stored candidate needs a string path list")
    required = ("id", "kind", "source_table")
    if any(not isinstance(data.get(field), str) for field in required):
        raise ValueError("stored candidate is missing id, kind, or source_table")
    attribute_table = data.get("attribute_table")
    attribute_column = data.get("attribute_column")
    if attribute_table is not None and not isinstance(attribute_table, str):
        raise ValueError("stored candidate attribute_table must be a string or null")
    if attribute_column is not None and not isinstance(attribute_column, str):
        raise ValueError("stored candidate attribute_column must be a string or null")
    return SignalCandidate(
        id=data["id"],  # type: ignore[arg-type]
        kind=data["kind"],  # type: ignore[arg-type]
        source_table=data["source_table"],  # type: ignore[arg-type]
        path=tuple(path),
        attribute_table=attribute_table,
        attribute_column=attribute_column,
    )


def enumerate_signal_candidates(
    dataset: RelationalDataset[object],
    graph: SchemaGraph,
    *,
    source_tables: Sequence[str] | None = None,
    max_hops: int = 2,
    include_relationships: bool = True,
    include_structural_columns: bool = False,
    catalog_view: CatalogView = "path-aware",
) -> tuple[SignalCandidate, ...]:
    """Return deterministic, schema-only signal candidates.

    ``path-aware`` (the default) gives each selected source table direct
    attributes and attributes reachable by at most ``max_hops`` foreign-key
    joins. A relational attribute remains distinct for each source/path pair:
    ``product -> review -> customer`` is a different usable signal from
    ``customer -> review -> product``. Paths do not repeat tables, following
    :meth:`SchemaGraph.find_paths`.

    ``feature-collapsed`` instead gives exactly one direct attribute candidate
    for each physical table/column. It intentionally omits relationship and
    path candidates, so it evaluates feature discovery without provenance.

    Identifiers and declared foreign-key columns are excluded by default. They
    describe database structure rather than a meaningful user-facing signal;
    relationship candidates preserve that structure explicitly.  Time columns
    remain eligible because recency and temporal behaviour may be relevant.
    """
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")
    if catalog_view not in CATALOG_VIEWS:
        raise ValueError(
            f"unknown catalog_view {catalog_view!r}; expected one of {CATALOG_VIEWS!r}"
        )

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
        paths: Iterable[list[str]] = (
            ([source_table],)
            if catalog_view == "feature-collapsed"
            else ([source_table], *graph.find_paths(source_table, max_hops))
        )
        for raw_path in paths:
            path = tuple(raw_path)
            if (
                catalog_view == "path-aware"
                and len(path) > 1
                and include_relationships
            ):
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


def _human_table_name(table: str, *, plural: bool) -> str:
    """Use a small schema-neutral pluralisation for compact path diagrams."""
    if not plural:
        return table
    if table.endswith("y") and len(table) > 1 and table[-2].lower() not in "aeiou":
        return f"{table[:-1]}ies"
    if table.endswith(("s", "x", "z", "ch", "sh")):
        return f"{table}es"
    return f"{table}s"


def _human_attribute_name(column: str, *, plural: bool) -> str:
    """Make a column label readable without relying on data values."""
    label = column.replace("_", " ")
    if not plural:
        return label
    if label.endswith("text"):
        return label
    if label.endswith("y") and len(label) > 1 and label[-2].lower() not in "aeiou":
        return f"{label[:-1]}ies"
    if label.endswith(("s", "x", "z", "ch", "sh")):
        return f"{label}es"
    return f"{label}s"


def _human_path_phrase(path: tuple[str, ...], graph: SchemaGraph) -> str:
    """Describe a short relational walk in terms useful to an annotator."""
    phrases: list[str] = []
    for source, target in zip(path, path[1:]):
        edge = _edge_for_step(graph, source, target)
        # Traversing from a referenced table to the table holding the foreign
        # key: one source row can have many target rows.
        if edge.target_table == source and edge.source_table == target:
            if target == "review":
                phrases.append("their reviews" if source == "customer" else "its reviews")
            else:
                phrases.append(f"related {_human_table_name(target, plural=True)}")
            continue
        # Traversing in foreign-key direction: one source row links to one
        # target row. The RelAmazon names make the common walks especially
        # understandable, with a schema-neutral fallback for other datasets.
        if source == "review" and target == "product":
            phrases.append("the products they reviewed")
        elif source == "review" and target == "customer":
            phrases.append("the customers who wrote them")
        else:
            phrases.append(f"the linked {_human_table_name(target, plural=True)}")
    return " to ".join(phrases)


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
