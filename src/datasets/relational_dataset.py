"""Storage-independent access to relational tables and their schema metadata.

Concrete dataset backends decide what a table object is and how it is loaded.
This module defines only the common API; it does not implement RelBench loading,
signal extraction, similarity, or benchmark generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from src.datasets.relbench_manifest import ForeignKey

__all__ = ["RelationalDataset"]

TableT = TypeVar("TableT")


class RelationalDataset(ABC, Generic[TableT]):
    """Minimal interface for relational data and its declared schema."""

    @abstractmethod
    def get_table(self, table_name: str) -> TableT:
        """Return one table in the concrete backend's native representation."""

    @abstractmethod
    def get_entity(self, table_name: str, entity_id: object) -> TableT:
        """Return the one-row table whose primary key equals ``entity_id``."""

    @abstractmethod
    def get_rows(self, table_name: str, column_name: str, value: object) -> TableT:
        """Return rows whose ``column_name`` equals ``value``."""

    @abstractmethod
    def get_tables(self) -> tuple[str, ...]:
        """Return the available table names."""

    def has_table(self, table_name: str) -> bool:
        """Return whether ``table_name`` is available."""
        return table_name in self.get_tables()

    @abstractmethod
    def get_primary_key(self, table_name: str) -> str | None:
        """Return the table's primary-key column, if one is declared."""

    @abstractmethod
    def get_foreign_keys(self) -> tuple[tuple[str, ForeignKey], ...]:
        """Return ``(source_table, foreign_key)`` for every declared foreign key."""

    @abstractmethod
    def get_time_column(self, table_name: str) -> str | None:
        """Return the table's time column, if one is declared."""
