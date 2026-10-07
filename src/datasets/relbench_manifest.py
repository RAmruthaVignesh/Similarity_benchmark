"""Parse a RelBench dataset ``manifest.yaml`` into a structured schema.

The manifest is the sole source of truth for a dataset's relational semantics:
primary keys, the foreign-key graph and time columns. Full column lists are not
part of it -- they live in the ``db/<table>.parquet`` files -- so only the
columns a manifest names are reported here. Nothing reads or downloads tables.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "DatasetSchema",
    "ForeignKey",
    "ManifestError",
    "TableSchema",
    "load_manifest",
    "parse_manifest",
]


class ManifestError(ValueError):
    """Raised when a manifest is malformed or missing a required key."""


@dataclass(frozen=True)
class ForeignKey:
    """``column`` of the owning table references the primary key of ``target_table``."""

    column: str
    target_table: str

    def to_dict(self) -> dict[str, str]:
        return {"column": self.column, "target_table": self.target_table}


@dataclass(frozen=True)
class TableSchema:
    """Relational semantics the manifest declares for one table."""

    name: str
    primary_key: str | None = None
    time_column: str | None = None
    foreign_keys: tuple[ForeignKey, ...] = ()
    columns: tuple[str, ...] | None = None

    @property
    def is_temporal(self) -> bool:
        return self.time_column is not None

    @property
    def foreign_key_columns(self) -> tuple[str, ...]:
        return tuple(fkey.column for fkey in self.foreign_keys)

    @property
    def referenced_columns(self) -> tuple[str, ...]:
        """Columns this manifest references, in primary-key / time / foreign-key order.

        A subset of the table's real columns, not an inventory of it: the rest of
        the columns exist only in ``db/<table>.parquet``.
        """
        ordered = [self.primary_key, self.time_column, *self.foreign_key_columns]
        return tuple(dict.fromkeys(col for col in ordered if col is not None))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "primary_key": self.primary_key,
            "time_column": self.time_column,
            "foreign_keys": [fkey.to_dict() for fkey in self.foreign_keys],
            "columns": None if self.columns is None else list(self.columns),
            "referenced_columns": list(self.referenced_columns),
        }


@dataclass(frozen=True)
class DatasetSchema:
    """A parsed dataset manifest: the table set, the foreign-key graph and the splits."""

    name: str
    tables: Mapping[str, TableSchema] = field(default_factory=dict)
    val_timestamp: str | None = None
    test_timestamp: str | None = None
    description: str | None = None
    manifest_version: int | None = None
    source: str | None = None

    @property
    def table_names(self) -> tuple[str, ...]:
        return tuple(self.tables)

    @property
    def primary_keys(self) -> dict[str, str | None]:
        return {name: table.primary_key for name, table in self.tables.items()}

    @property
    def temporal_columns(self) -> dict[str, str]:
        """Time column per table, for the tables that have one."""
        return {
            name: table.time_column
            for name, table in self.tables.items()
            if table.time_column is not None
        }

    def foreign_key_edges(self) -> tuple[tuple[str, str, str], ...]:
        """``(table, column, target_table)`` for every foreign key in the dataset."""
        return tuple(
            (name, fkey.column, fkey.target_table)
            for name, table in self.tables.items()
            for fkey in table.foreign_keys
        )

    def dangling_foreign_keys(self) -> tuple[tuple[str, str, str], ...]:
        """Foreign-key edges whose target table the manifest does not declare."""
        return tuple(
            edge for edge in self.foreign_key_edges() if edge[2] not in self.tables
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "manifest_version": self.manifest_version,
            "description": self.description,
            "val_timestamp": self.val_timestamp,
            "test_timestamp": self.test_timestamp,
            "source": self.source,
            "tables": {name: table.to_dict() for name, table in self.tables.items()},
            "foreign_key_edges": [list(edge) for edge in self.foreign_key_edges()],
            "temporal_columns": self.temporal_columns,
        }


def load_manifest(path: str | Path) -> DatasetSchema:
    """Parse the manifest at ``path``."""
    path = Path(path)
    with open(path) as handle:
        data = yaml.safe_load(handle)
    return parse_manifest(data, source=str(path))


def parse_manifest(data: Any, *, source: str | None = None) -> DatasetSchema:
    """Parse an already-loaded manifest mapping."""
    if not isinstance(data, Mapping):
        raise ManifestError(
            f"manifest must be a mapping, got {type(data).__name__}"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ManifestError("manifest is missing a 'name'")

    raw_tables = data.get("tables")
    if raw_tables is None:
        raise ManifestError(f"manifest '{name}' is missing 'tables'")
    if not isinstance(raw_tables, Mapping):
        raise ManifestError(
            f"manifest '{name}': 'tables' must be a mapping of table name to spec"
        )

    return DatasetSchema(
        name=name,
        tables={
            table_name: _parse_table(table_name, spec)
            for table_name, spec in raw_tables.items()
        },
        val_timestamp=_parse_timestamp(data.get("val_timestamp")),
        test_timestamp=_parse_timestamp(data.get("test_timestamp")),
        description=data.get("description"),
        manifest_version=data.get("manifest_version"),
        source=source,
    )


def _parse_table(name: str, spec: Any) -> TableSchema:
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise ManifestError(
            f"table '{name}': spec must be a mapping, got {type(spec).__name__}"
        )

    raw_fkeys = spec.get("fkeys") or {}
    if not isinstance(raw_fkeys, Mapping):
        raise ManifestError(
            f"table '{name}': 'fkeys' must map a foreign-key column to a table name"
        )

    return TableSchema(
        name=name,
        primary_key=spec.get("pkey"),
        time_column=spec.get("time_col"),
        foreign_keys=tuple(
            ForeignKey(column=str(column), target_table=str(target))
            for column, target in raw_fkeys.items()
        ),
        columns=_parse_columns(name, spec.get("columns")),
    )


def _parse_columns(table: str, raw: Any) -> tuple[str, ...] | None:
    """Declared column names, or ``None``: v1 manifests do not list columns."""
    if raw is None:
        return None
    if isinstance(raw, Mapping):  # column name -> dtype
        return tuple(str(column) for column in raw)
    if isinstance(raw, (list, tuple)):
        names = []
        for entry in raw:
            if isinstance(entry, Mapping):
                column = entry.get("name")
                if column is None:
                    raise ManifestError(
                        f"table '{table}': column entry {entry!r} has no 'name'"
                    )
                names.append(str(column))
            else:
                names.append(str(entry))
        return tuple(names)
    raise ManifestError(f"table '{table}': 'columns' must be a list or a mapping")


def _parse_timestamp(value: Any) -> str | None:
    """Split timestamps as ISO strings; yaml may have parsed them into dates already."""
    if value is None:
        return None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    return str(value)
