"""Configuration for a small local development subset of a RelBench dataset.

This fixes only *how big* a subset is and *where* its files belong: one parquet
file per table plus a manifest, under ``data/dev/<dataset>/``. The table names
are not listed here -- they come from the dataset's manifest, which stays the
single source of truth. Nothing here downloads, reads or samples any data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_MAX_REVIEWS_PER_PRODUCT",
    "DEFAULT_MIN_REVIEWS_PER_PRODUCT",
    "DEFAULT_NUM_PRODUCTS",
    "DEFAULT_REVIEW_ROW_GROUPS",
    "DEFAULT_SAMPLING_MODE",
    "DEFAULT_SEED",
    "DEV_ROOT",
    "ENTITY_TABLE",
    "LINK_TABLE",
    "SAMPLING_MODES",
    "DevSubsetConfig",
    "DevSubsetError",
]

DEV_ROOT = Path("data/dev")
DEFAULT_DATASET = "rel-amazon"
DEFAULT_NUM_PRODUCTS = 500
DEFAULT_MIN_REVIEWS_PER_PRODUCT = 20
DEFAULT_MAX_REVIEWS_PER_PRODUCT = 100
DEFAULT_REVIEW_ROW_GROUPS = 10
# The table whose rows ``num_products`` and the per-product limits count.
ENTITY_TABLE = "product"
# The table through which overlap-aware sampling connects entities.
LINK_TABLE = "customer"
# "entity_centered" picks eligible products pseudo-randomly; "overlap_aware"
# grows a set of products that share customers. Both are for development only.
SAMPLING_MODES = ("entity_centered", "overlap_aware")
DEFAULT_SAMPLING_MODE = "entity_centered"
DEFAULT_SEED = 0
MANIFEST_NAME = "manifest.yaml"


class DevSubsetError(ValueError):
    """Raised when a development-subset configuration is not usable."""


@dataclass(frozen=True)
class DevSubsetConfig:
    """How large a development subset is and where it is written.

    The sample is centred on products: up to ``num_products`` products with at
    least ``min_reviews_per_product`` reviews, each keeping at most
    ``max_reviews_per_product`` of them; customers follow from the reviews kept.
    ``sampling_mode`` decides which eligible products: pseudo-randomly
    (``"entity_centered"``) or as a set that shares customers
    (``"overlap_aware"``). Reviews are drawn from ``review_row_groups`` evenly spaced parquet row
    groups of the source file, and ``seed`` fixes which products and reviews
    are chosen. ``output_dir`` accepts a string or a ``Path`` and defaults to
    ``data/dev/<dataset>``; it is always a ``Path`` once constructed.
    """

    dataset: str = DEFAULT_DATASET
    num_products: int = DEFAULT_NUM_PRODUCTS
    min_reviews_per_product: int = DEFAULT_MIN_REVIEWS_PER_PRODUCT
    max_reviews_per_product: int = DEFAULT_MAX_REVIEWS_PER_PRODUCT
    seed: int = DEFAULT_SEED
    review_row_groups: int = DEFAULT_REVIEW_ROW_GROUPS
    sampling_mode: str = DEFAULT_SAMPLING_MODE
    output_dir: Path | str | None = None

    def __post_init__(self) -> None:
        if not self.dataset:
            raise DevSubsetError("dataset name must not be empty")
        if self.sampling_mode not in SAMPLING_MODES:
            raise DevSubsetError(
                f"sampling_mode must be one of {SAMPLING_MODES}, "
                f"got {self.sampling_mode!r}"
            )
        for name in ("num_products", "min_reviews_per_product"):
            if getattr(self, name) < 1:
                raise DevSubsetError(
                    f"{name} must be at least 1, got {getattr(self, name)}"
                )
        if self.max_reviews_per_product < self.min_reviews_per_product:
            raise DevSubsetError(
                f"max_reviews_per_product ({self.max_reviews_per_product}) must be "
                f"at least min_reviews_per_product ({self.min_reviews_per_product})"
            )
        if self.review_row_groups < 1:
            raise DevSubsetError(
                f"review_row_groups must be at least 1, got {self.review_row_groups}"
            )
        resolved = (
            DEV_ROOT / self.dataset
            if self.output_dir is None
            else Path(self.output_dir)
        )
        object.__setattr__(self, "output_dir", resolved)

    @property
    def review_rows(self) -> int:
        """The most reviews the subset can hold."""
        return self.num_products * self.max_reviews_per_product

    @property
    def manifest_path(self) -> Path:
        """The subset's own manifest, written beside its tables."""
        return self.output_dir / MANIFEST_NAME

    def table_path(self, table_name: str) -> Path:
        """Where the subset's copy of ``table_name`` belongs."""
        return self.output_dir / f"{table_name}.parquet"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "num_products": self.num_products,
            "min_reviews_per_product": self.min_reviews_per_product,
            "max_reviews_per_product": self.max_reviews_per_product,
            "review_rows": self.review_rows,
            "seed": self.seed,
            "review_row_groups": self.review_row_groups,
            "sampling_mode": self.sampling_mode,
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
        }
