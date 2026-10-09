"""Local Parquet implementation of the generic relational dataset interface.

The manifest is parsed when the dataset is constructed, but Parquet tables are
loaded only when requested. This module does not build subsets or implement
signals, similarity, or benchmark generation.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.datasets.relational_dataset import RelationalDataset
from src.datasets.relbench_manifest import (
    DatasetSchema,
    ForeignKey,
    TableSchema,
    load_manifest,
)

__all__ = ["DEFAULT_SUBSET_ROOT", "LocalRelBenchDataset"]

DEFAULT_SUBSET_ROOT = Path("data/dev/rel-amazon")


class LocalRelBenchDataset(RelationalDataset[pa.Table]):
    """A manifest-described RelBench subset stored as local Parquet files."""

    def __init__(self, root: str | Path = DEFAULT_SUBSET_ROOT) -> None:
        self.root = Path(root)
        self.schema: DatasetSchema = load_manifest(self.root / "manifest.yaml")
        self._tables: dict[str, pa.Table] = {}

    def get_table(self, table_name: str) -> pa.Table:
        """Load and cache one local Parquet table."""
        self._require_table(table_name)
        if table_name not in self._tables:
            self._tables[table_name] = pq.read_table(
                self.root / f"{table_name}.parquet"
            )
        return self._tables[table_name]

    def get_entity(self, table_name: str, entity_id: object) -> pa.Table:
        """Return the one row matching the table's declared primary key."""
        primary_key = self.get_primary_key(table_name)
        if primary_key is None:
            raise ValueError(f"table '{table_name}' declares no primary key")
        rows = self.get_rows(table_name, primary_key, entity_id)
        if rows.num_rows == 0:
            raise KeyError(
                f"table '{table_name}' has no entity with "
                f"{primary_key}={entity_id!r}"
            )
        if rows.num_rows > 1:
            raise ValueError(
                f"table '{table_name}' has {rows.num_rows} rows with "
                f"{primary_key}={entity_id!r}; primary keys must be unique"
            )
        return rows

    def get_rows(self, table_name: str, column_name: str, value: object) -> pa.Table:
        """Return rows whose named column equals ``value``."""
        table = self.get_table(table_name)
        if column_name not in table.column_names:
            raise KeyError(
                f"unknown column '{column_name}' in table '{table_name}'; "
                f"available columns: {', '.join(table.column_names) or 'none'}"
            )
        return table.filter(pc.equal(table[column_name], value))

    def get_tables(self) -> tuple[str, ...]:
        """Return table names in manifest order without loading their data."""
        return self.schema.table_names

    def get_primary_key(self, table_name: str) -> str | None:
        """Return the primary-key column declared by the manifest."""
        return self._table_schema(table_name).primary_key

    def get_foreign_keys(self) -> tuple[tuple[str, ForeignKey], ...]:
        """Return every manifest foreign key together with its source table."""
        return tuple(
            (table.name, foreign_key)
            for table in self.schema.tables.values()
            for foreign_key in table.foreign_keys
        )

    def get_time_column(self, table_name: str) -> str | None:
        """Return the time column declared by the manifest."""
        return self._table_schema(table_name).time_column

    def _table_schema(self, table_name: str) -> TableSchema:
        self._require_table(table_name)
        return self.schema.tables[table_name]

    def _require_table(self, table_name: str) -> None:
        if table_name not in self.schema.tables:
            raise KeyError(
                f"unknown table '{table_name}'; manifest '{self.schema.name}' "
                f"declares {', '.join(self.schema.table_names) or 'none'}"
            )
