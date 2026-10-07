"""Build a small, referentially closed development subset of a RelBench dataset.

The fact table -- the one table the manifest gives foreign keys, ``review`` for
rel-amazon -- is sampled around a set of products from the remote parquet with
DuckDB, without downloading the whole multi-GB file: only a spread of its row
groups is read (see :func:`_sample_fact_query`). Each table it references is
then filtered to the keys the sample actually uses, so every foreign key in the
subset resolves. Table names, key columns and join directions all come from the
manifest.
"""

from __future__ import annotations

import heapq
import itertools
import shutil
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from pathlib import Path

import duckdb
import pyarrow as pa

from src.datasets.dev_subset import ENTITY_TABLE, LINK_TABLE, DevSubsetConfig
from src.datasets.relbench_manifest import load_manifest
from src.schema.schema_graph import Edge, SchemaGraph

__all__ = [
    "RELBENCH_BASE_URL",
    "SubsetBuildError",
    "build_dev_subset",
    "plan_subset",
    "remote_table_url",
]

RELBENCH_BASE_URL = (
    "https://huggingface.co/datasets/stanford-star/relbench-v1/resolve/main"
)


class SubsetBuildError(ValueError):
    """Raised when a manifest's layout is not one this builder can subset safely."""


def remote_table_url(
    dataset: str, table: str, base_url: str = RELBENCH_BASE_URL
) -> str:
    """Where RelBench hosts ``table`` of ``dataset``."""
    return f"{base_url}/{dataset}/db/{table}.parquet"


def build_dev_subset(
    config: DevSubsetConfig,
    manifest_path: str | Path,
    *,
    base_url: str = RELBENCH_BASE_URL,
) -> dict[str, int]:
    """Write the subset ``config`` describes and return the rows written per table.

    The fact-table sample is centred on ``ENTITY_TABLE`` (see
    :func:`_sample_fact_query`) and hash-selected using ``config.seed``, so
    reruns with the same config reproduce it exactly. The source manifest is
    copied in beside the tables.
    """
    manifest = load_manifest(manifest_path)
    if manifest.name != config.dataset:
        raise SubsetBuildError(
            f"manifest describes '{manifest.name}', config asks for '{config.dataset}'"
        )
    graph = SchemaGraph.from_manifest(manifest)
    fact_table, references = plan_subset(graph)
    entity_edge = graph.get_relationship(fact_table, ENTITY_TABLE)
    if entity_edge is None:
        raise SubsetBuildError(
            f"'{fact_table}' has no foreign key to '{ENTITY_TABLE}' to sample by"
        )
    link_column = None
    if config.sampling_mode == "overlap_aware":
        link_edge = graph.get_relationship(fact_table, LINK_TABLE)
        if link_edge is None:
            raise SubsetBuildError(
                f"'{fact_table}' has no foreign key to '{LINK_TABLE}' to connect by"
            )
        link_column = link_edge.source_column

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    def url(table: str) -> str:
        return _literal(remote_table_url(config.dataset, table, base_url))

    fact_path = config.table_path(fact_table)
    _write(
        con,
        _sample_fact_query(
            con, url(fact_table), entity_edge.source_column, config, link_column
        ),
        fact_path,
    )
    rows = {fact_table: _count(con, fact_path)}

    for target_table, edges in references.items():
        used_keys = " UNION ".join(
            f"SELECT {_ident(edge.source_column)} "
            f"FROM read_parquet({_literal(str(fact_path))})"
            for edge in edges
        )
        target_path = config.table_path(target_table)
        _write(
            con,
            f"SELECT * FROM read_parquet({url(target_table)}) "
            f"WHERE {_ident(edges[0].target_column)} IN ({used_keys})",
            target_path,
        )
        rows[target_table] = _count(con, target_path)

    shutil.copyfile(manifest_path, config.manifest_path)
    return rows


def plan_subset(graph: SchemaGraph) -> tuple[str, dict[str, list[Edge]]]:
    """The fact table, and the foreign keys into each table it references.

    Only a star layout is closed by one filtering pass: a single table holding
    every foreign key, pointing at tables that hold none.
    """
    sources = {edge.source_table for edge in graph.get_edges()}
    if len(sources) != 1:
        raise SubsetBuildError(
            f"expected exactly one table with foreign keys, found "
            f"{sorted(sources) or 'none'}"
        )
    fact_table = sources.pop()

    references: dict[str, list[Edge]] = {}
    for edge in graph.get_edges():
        references.setdefault(edge.target_table, []).append(edge)
    if fact_table in references:
        raise SubsetBuildError(f"table '{fact_table}' references itself")

    unreferenced = [
        table.name
        for table in graph.get_tables()
        if table.name != fact_table and table.name not in references
    ]
    if unreferenced:
        raise SubsetBuildError(
            f"tables {unreferenced} are not referenced by '{fact_table}'"
        )
    return fact_table, references


def _sample_fact_query(
    con: duckdb.DuckDBPyConnection,
    source: str,
    entity_column: str,
    config: DevSubsetConfig,
    link_column: str | None = None,
) -> str:
    """A query for a deterministic, entity-centred sample of the fact table.

    Up to ``config.num_products`` entities with at least
    ``config.min_reviews_per_product`` fact rows are chosen, and each keeps at
    most ``config.max_reviews_per_product`` of them. Counts are taken over the
    same rows the sample is drawn from, so every chosen entity meets the minimum
    before truncation. Without ``link_column`` the eligible entities are chosen
    pseudo-randomly; with it they are grown as a set sharing link values (see
    :func:`_choose_connected_entities`), and truncation keeps shared links first.

    DuckDB fetches a remote parquet file one whole row group at a time, and only
    the columns a query uses. So the sample is drawn from evenly spaced row
    groups -- which spreads it across a time-sorted file such as rel-amazon's
    ``review`` -- and chosen from the entity column alone: counting rows per
    entity reads one integer column, and only the final fetch reads whole rows.
    Entities and rows are ranked by an md5 of their key or position and the
    seed, so the choice is pseudo-random but identical on every run.
    """
    sizes = [
        rows
        for (rows,) in con.execute(
            f"SELECT any_value(row_group_num_rows) FROM parquet_metadata({source}) "
            f"GROUP BY row_group_id ORDER BY row_group_id"
        ).fetchall()
    ]
    starts = list(itertools.accumulate([0, *sizes[:-1]]))
    groups = [
        (starts[group], starts[group] + sizes[group])
        for group in _evenly_spaced(len(sizes), config.review_row_groups)
    ]

    def scan(columns: str) -> str:
        # A constant range per scan lets DuckDB skip every row group outside it.
        return " UNION ALL ".join(
            f"SELECT {columns} FROM read_parquet({source}, file_row_number = true) "
            f"WHERE file_row_number BETWEEN {start} AND {end - 1}"
            for start, end in groups
        )

    entity = _ident(entity_column)
    seed = int(config.seed)
    columns = f"file_row_number AS position, {entity} AS entity"
    if link_column is not None:
        columns += f", {_ident(link_column)} AS link"
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE pool AS "
        f"SELECT * FROM ({scan(columns)}) WHERE entity IS NOT NULL"
    )
    eligible = (
        f"SELECT entity FROM pool GROUP BY entity "
        f"HAVING count(*) >= {int(config.min_reviews_per_product)}"
    )
    if link_column is None:
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE chosen_entities AS {eligible} "
            f"ORDER BY md5_number(concat({seed}, ':', entity)), entity "
            f"LIMIT {int(config.num_products)}"
        )
        ranked_from, first_key = "pool AS p", ""
    else:
        _choose_connected_entities(con, eligible, config)
        con.execute(
            "CREATE OR REPLACE TEMP TABLE shared_links AS "
            "SELECT link FROM pool "
            "WHERE link IS NOT NULL AND entity IN (SELECT entity FROM chosen_entities) "
            "GROUP BY link HAVING count(DISTINCT entity) >= 2"
        )
        # Truncation keeps reviews by shared customers first, so it keeps the overlap.
        ranked_from = "pool AS p LEFT JOIN shared_links AS s ON p.link = s.link"
        first_key = "s.link IS NULL, "
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE sampled_positions AS "
        f"SELECT p.position FROM {ranked_from} "
        f"WHERE p.entity IN (SELECT entity FROM chosen_entities) "
        f"QUALIFY row_number() OVER (PARTITION BY p.entity "
        f"ORDER BY {first_key}md5_number(concat({seed}, ':', p.position)), p.position) "
        f"<= {int(config.max_reviews_per_product)}"
    )
    return (
        f"SELECT * EXCLUDE (file_row_number) FROM ({scan('*')}) "
        f"WHERE file_row_number IN (SELECT position FROM sampled_positions) "
        f"ORDER BY file_row_number"
    )


def _choose_connected_entities(
    con: duckdb.DuckDBPyConnection, eligible: str, config: DevSubsetConfig
) -> None:
    """Fill ``chosen_entities`` with eligible entities that share many links.

    Only links joining at least two eligible entities can connect anything, so
    only those leave DuckDB. Seeds are tried most-connected first, with ties
    broken by an md5 of the entity and ``config.seed``.
    """
    seed = int(config.seed)
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE bridges AS "
        f"SELECT entity, link FROM (SELECT DISTINCT entity, link FROM pool "
        f"WHERE link IS NOT NULL AND entity IN ({eligible})) "
        f"QUALIFY count(*) OVER (PARTITION BY link) >= 2"
    )
    order = con.execute(
        f"SELECT e.entity FROM ({eligible}) AS e "
        f"LEFT JOIN (SELECT entity, count(*) AS n FROM bridges GROUP BY entity) AS b "
        f"USING (entity) "
        f"ORDER BY coalesce(b.n, 0) DESC, "
        f"md5_number(concat({seed}, ':', e.entity)), e.entity"
    ).to_arrow_table()
    bridges = con.execute(
        "SELECT entity, link FROM bridges ORDER BY entity, link"
    ).to_arrow_table()
    selected = _connected_selection(
        order["entity"].to_pylist(),
        zip(bridges["entity"].to_pylist(), bridges["link"].to_pylist()),
        config.num_products,
    )
    con.register(
        "selected_entities",
        pa.table({"entity": pa.array(selected, type=order["entity"].type)}),
    )
    con.execute(
        "CREATE OR REPLACE TEMP TABLE chosen_entities AS "
        "SELECT entity FROM selected_entities"
    )
    con.unregister("selected_entities")


def _connected_selection(
    seed_order: Sequence[Hashable],
    links: Iterable[tuple[Hashable, Hashable]],
    limit: int,
) -> list[Hashable]:
    """Up to ``limit`` entities, grown greedily through shared links.

    Starting from the first entity in ``seed_order``, the next pick is always
    the unpicked entity sharing the most distinct links with those picked so
    far (ties go to the earlier one in ``seed_order``). When nothing picked
    shares a link with anything left, the next unpicked entity in
    ``seed_order`` starts a new expansion. ``links`` are (entity, link) pairs;
    entities outside ``seed_order`` are never picked.
    """
    rank = {entity: i for i, entity in enumerate(seed_order)}
    links_of: dict[Hashable, list[Hashable]] = defaultdict(list)
    entities_of: dict[Hashable, list[Hashable]] = defaultdict(list)
    for entity, link in links:
        if entity in rank:
            links_of[entity].append(link)
            entities_of[link].append(entity)

    selected: list[Hashable] = []
    picked: set[Hashable] = set()
    reached: set[Hashable] = set()
    shared: dict[Hashable, int] = defaultdict(int)
    frontier: list[tuple[int, int, Hashable]] = []
    seeds = iter(seed_order)
    while len(selected) < limit:
        entity = None
        while frontier:
            negative, _, candidate = heapq.heappop(frontier)
            if candidate not in picked and -negative == shared[candidate]:
                entity = candidate
                break
        if entity is None:
            entity = next((e for e in seeds if e not in picked), None)
            if entity is None:
                break
        picked.add(entity)
        selected.append(entity)
        for link in links_of[entity]:
            if link in reached:
                continue
            reached.add(link)
            for other in entities_of[link]:
                if other not in picked:
                    shared[other] += 1
                    heapq.heappush(frontier, (-shared[other], rank[other], other))
    return selected


def _evenly_spaced(count: int, chosen: int) -> list[int]:
    """``chosen`` indices out of ``range(count)``, evenly spread, both ends included."""
    if chosen >= count:
        return list(range(count))
    if chosen == 1:
        return [(count - 1) // 2]
    return sorted({round(i * (count - 1) / (chosen - 1)) for i in range(chosen)})


def _write(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    con.execute(f"COPY ({query}) TO {_literal(str(path))} (FORMAT parquet)")


def _count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return con.execute(
        f"SELECT count(*) FROM read_parquet({_literal(str(path))})"
    ).fetchone()[0]


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
