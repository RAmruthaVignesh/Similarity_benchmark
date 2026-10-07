#!/usr/bin/env python3
"""Print the schema parsed from a local RelBench dataset manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets.relbench_manifest import (  # noqa: E402
    DatasetSchema,
    ManifestError,
    load_manifest,
)

DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifests" / "rel-amazon" / "manifest.yaml"

_LABEL_WIDTH = len("manifest-referenced columns:") + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"path to a dataset manifest.yaml (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the parsed schema as JSON instead of a report",
    )
    args = parser.parse_args()

    try:
        schema = load_manifest(args.manifest)
    except (OSError, ManifestError) as error:
        parser.exit(2, f"error: {error}\n")

    if args.json:
        print(json.dumps(schema.to_dict(), indent=2))
    else:
        print_report(schema)
    return 0


def print_report(schema: DatasetSchema) -> None:
    version = schema.manifest_version
    print(f"{schema.name}" + (f"  (manifest v{version})" if version else ""))
    if schema.description:
        print(f"  {schema.description}")
    print(f"  val_timestamp:  {schema.val_timestamp or '-'}")
    print(f"  test_timestamp: {schema.test_timestamp or '-'}")

    print(f"\ntables ({len(schema.tables)})")
    for table in schema.tables.values():
        print(f"  {table.name}")
        print(_field("primary key", table.primary_key))
        print(_field("time column", table.time_column))
        print(
            _field(
                "foreign keys",
                ", ".join(
                    f"{fkey.column} -> {fkey.target_table}"
                    for fkey in table.foreign_keys
                ),
            )
        )
        if table.columns is None:
            print(
                _field(
                    "manifest-referenced columns",
                    ", ".join(table.referenced_columns),
                )
            )
        else:
            print(_field("columns", ", ".join(table.columns)))

    edges = schema.foreign_key_edges()
    print(f"\nforeign-key graph ({len(edges)} edges)")
    for table, column, target in edges:
        print(f"  {table}.{column} -> {target}")
    if not edges:
        print("  -")

    temporal = schema.temporal_columns
    print(f"\ntemporal columns ({len(temporal)} of {len(schema.tables)} tables)")
    for table, column in temporal.items():
        print(f"  {table}.{column}")
    if not temporal:
        print("  -")

    dangling = schema.dangling_foreign_keys()
    if dangling:
        print("\nwarning: foreign keys pointing at undeclared tables")
        for table, column, target in dangling:
            print(f"  {table}.{column} -> {target}")

    if all(table.columns is None for table in schema.tables.values()):
        print(
            "\nnote: this manifest declares no column lists, so the columns above are "
            "only the ones it references as keys or time columns. Each table has "
            "further attributes that exist only in db/<table>.parquet, which is not "
            "read here."
        )


def _field(label: str, value: str | None) -> str:
    return f"    {label + ':':<{_LABEL_WIDTH}}{value or '-'}"


if __name__ == "__main__":
    raise SystemExit(main())
