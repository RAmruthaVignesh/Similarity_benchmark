"""Check that a local development subset is complete and referentially closed.

Every check is derived from the subset's own ``manifest.yaml``: which tables
must exist, which columns are primary keys and which foreign keys must resolve.
Results are collected rather than raised, so a single run reports every failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from src.datasets.dev_subset import ENTITY_TABLE, LINK_TABLE, DevSubsetConfig
from src.datasets.dev_subset_builder import _ident, _literal, plan_subset
from src.datasets.relbench_manifest import load_manifest
from src.schema.schema_graph import SchemaGraph

__all__ = [
    "Check",
    "CountSummary",
    "OverlapSummary",
    "SubsetReport",
    "validate_dev_subset",
]


@dataclass(frozen=True)
class CountSummary:
    """How many fact rows each referenced key has, across the keys that have any."""

    min: int
    median: float
    mean: float
    max: int


@dataclass(frozen=True)
class OverlapSummary:
    """How the subset's entities share link values, e.g. products sharing customers.

    A pair overlaps when both entities have at least one fact row with the same
    link value; ``shared_*`` and ``median_jaccard`` (of the two link sets) cover
    overlapping pairs only and are ``None`` when there are none.
    """

    entity_table: str
    link_table: str
    entities: int
    connected_entities: int
    overlapping_pairs: int
    bridging_links: int
    shared_median: float | None
    shared_mean: float | None
    shared_max: int | None
    median_jaccard: float | None

    @property
    def possible_pairs(self) -> int:
        return self.entities * (self.entities - 1) // 2

    @property
    def overlapping_pair_fraction(self) -> float:
        return self.overlapping_pairs / self.possible_pairs if self.possible_pairs else 0.0


@dataclass(frozen=True)
class Check:
    """One validation check and its outcome."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class SubsetReport:
    """Per-table statistics and the outcome of every check that could run."""

    dataset: str
    fact_table: str | None = None
    row_counts: dict[str, int] = field(default_factory=dict)
    distinct_keys: dict[str, int] = field(default_factory=dict)
    rows_per_key: dict[str, CountSummary] = field(default_factory=dict)
    overlap: OverlapSummary | None = None
    time_column: str | None = None
    time_range: tuple[Any, Any] | None = None
    checks: tuple[Check, ...] = ()

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.passed)

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_dev_subset(config: DevSubsetConfig) -> SubsetReport:
    """Validate the subset under ``config.output_dir`` against its own manifest.

    Checks that depend on a missing table are skipped; the missing table is
    already reported as a failure.
    """
    if not Path(config.manifest_path).exists():
        return SubsetReport(
            dataset=config.dataset,
            checks=(Check("manifest exists", False, str(config.manifest_path)),),
        )
    manifest = load_manifest(config.manifest_path)
    graph = SchemaGraph.from_manifest(manifest)
    fact_table, _ = plan_subset(graph)
    con = duckdb.connect()
    checks = [Check("manifest exists", True)]

    scans = {}
    for table in graph.get_tables():
        path = config.table_path(table.name)
        exists = path.exists()
        checks.append(Check(f"table {table.name} exists", exists, "" if exists else str(path)))
        if exists:
            scans[table.name] = f"read_parquet({_literal(str(path))})"

    row_counts = {}
    for name, scan in scans.items():
        row_counts[name] = con.execute(f"SELECT count(*) FROM {scan}").fetchone()[0]
        checks.append(Check(f"table {name} is not empty", row_counts[name] > 0))

    if fact_table in row_counts:
        checks.append(
            Check(
                f"{fact_table} has at most {config.review_rows:,} rows",
                row_counts[fact_table] <= config.review_rows,
                f"{row_counts[fact_table]:,} rows",
            )
        )

    distinct_keys = {}
    for table in graph.get_tables():
        if table.primary_key is None or table.name not in scans:
            continue
        key = _ident(table.primary_key)
        non_null, distinct = con.execute(
            f"SELECT count({key}), count(DISTINCT {key}) FROM {scans[table.name]}"
        ).fetchone()
        distinct_keys[table.name] = distinct
        duplicates = non_null - distinct
        checks.append(
            Check(
                f"{table.name}.{table.primary_key} is unique",
                duplicates == 0,
                f"{duplicates:,} duplicate values" if duplicates else "",
            )
        )

    for edge in graph.get_edges():
        if edge.source_table not in scans or edge.target_table not in scans:
            continue
        source, target = _ident(edge.source_column), _ident(edge.target_column)
        dangling = con.execute(
            f"SELECT count(*) FROM {scans[edge.source_table]} AS s "
            f"WHERE s.{source} IS NOT NULL AND NOT EXISTS ("
            f"SELECT 1 FROM {scans[edge.target_table]} AS t "
            f"WHERE t.{target} = s.{source})"
        ).fetchone()[0]
        checks.append(
            Check(
                f"{edge.source_table}.{edge.source_column} -> "
                f"{edge.target_table}.{edge.target_column}",
                dangling == 0,
                f"{dangling:,} rows without a match" if dangling else "",
            )
        )

    rows_per_key = {}
    for edge in graph.get_edges():
        if edge.source_table not in scans:
            continue
        column = _ident(edge.source_column)
        low, median, mean, high = con.execute(
            f"SELECT min(n), median(n), avg(n), max(n) FROM ("
            f"SELECT count(*) AS n FROM {scans[edge.source_table]} "
            f"WHERE {column} IS NOT NULL GROUP BY {column})"
        ).fetchone()
        if low is not None:
            rows_per_key[edge.target_table] = CountSummary(low, median, mean, high)

    time_column = manifest.tables[fact_table].time_column
    time_range = None
    if time_column is not None and row_counts.get(fact_table):
        column = _ident(time_column)
        time_range = con.execute(
            f"SELECT min({column}), max({column}) FROM {scans[fact_table]}"
        ).fetchone()

    return SubsetReport(
        dataset=manifest.name,
        fact_table=fact_table,
        row_counts=row_counts,
        distinct_keys=distinct_keys,
        rows_per_key=rows_per_key,
        overlap=(
            _overlap(con, graph, fact_table, scans[fact_table])
            if fact_table in scans
            else None
        ),
        time_column=time_column,
        time_range=time_range,
        checks=tuple(checks),
    )


def _overlap(
    con: duckdb.DuckDBPyConnection, graph: SchemaGraph, fact_table: str, scan: str
) -> OverlapSummary | None:
    """How ``ENTITY_TABLE`` rows share ``LINK_TABLE`` keys through the fact table."""
    entity_edge = graph.get_relationship(fact_table, ENTITY_TABLE)
    link_edge = graph.get_relationship(fact_table, LINK_TABLE)
    if entity_edge is None or link_edge is None:
        return None
    entity, link = _ident(entity_edge.source_column), _ident(link_edge.source_column)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE entity_links AS "
        f"SELECT DISTINCT {entity} AS entity, {link} AS link FROM {scan} "
        f"WHERE {entity} IS NOT NULL AND {link} IS NOT NULL"
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE entity_pairs AS "
        "SELECT a.entity AS left_entity, b.entity AS right_entity, count(*) AS shared "
        "FROM entity_links AS a JOIN entity_links AS b "
        "ON a.link = b.link AND a.entity < b.entity GROUP BY ALL"
    )
    entities = con.execute(
        f"SELECT count(DISTINCT {entity}) FROM {scan}"
    ).fetchone()[0]
    connected = con.execute(
        "SELECT count(*) FROM "
        "(SELECT left_entity FROM entity_pairs UNION SELECT right_entity FROM entity_pairs)"
    ).fetchone()[0]
    bridging = con.execute(
        "SELECT count(*) FROM (SELECT link FROM entity_links "
        "GROUP BY link HAVING count(*) >= 2)"
    ).fetchone()[0]
    pairs, median, mean, high, jaccard = con.execute(
        "WITH sizes AS (SELECT entity, count(*) AS n FROM entity_links GROUP BY entity) "
        "SELECT count(*), median(shared), avg(shared), max(shared), "
        "median(shared / (f.n + s.n - shared)) FROM entity_pairs "
        "JOIN sizes AS f ON f.entity = left_entity JOIN sizes AS s ON s.entity = right_entity"
    ).fetchone()
    return OverlapSummary(
        entity_table=ENTITY_TABLE,
        link_table=LINK_TABLE,
        entities=entities,
        connected_entities=connected,
        overlapping_pairs=pairs,
        bridging_links=bridging,
        shared_median=median,
        shared_mean=mean,
        shared_max=high,
        median_jaccard=jaccard,
    )
