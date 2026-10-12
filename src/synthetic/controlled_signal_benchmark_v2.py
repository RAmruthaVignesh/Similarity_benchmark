"""Controlled qrels on the real RelAmazon schema and signal candidate pool.

Only the schema manifest and Parquet schemas are copied into the benchmark;
source records are never copied or used. The qrels remain deterministic and
author-defined so this is a controlled capability check, not human ground truth.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets.local_relbench_dataset import LocalRelBenchDataset
from src.schema.schema_graph import SchemaGraph
from src.signals.candidates import SignalCandidate, enumerate_signal_candidates

BENCHMARK_VERSION = "controlled-rel-amazon-signals-2"
MAX_HOPS = 2
EXPECTED_CANDIDATE_COUNT = 39
DEFAULT_SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "data" / "dev" / "rel-amazon"


@dataclass(frozen=True)
class ControlledSignalTask:
    id: str
    condition: str
    source_table: str
    reasoning_type: str
    oracle_rationale: str
    signal_grades: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "condition": self.condition,
            "source_table": self.source_table,
            "reasoning_type": self.reasoning_type,
            "oracle_rationale": self.oracle_rationale,
            "signal_grades": self.signal_grades,
        }


def build_controlled_signal_benchmark(
    root: str | Path,
    *,
    schema_root: str | Path = DEFAULT_SCHEMA_ROOT,
) -> tuple[ControlledSignalTask, ...]:
    """Create a portable schema-only snapshot and deterministic synthetic qrels."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _write_schema_only_snapshot(Path(schema_root), root)

    dataset = LocalRelBenchDataset(root)
    graph = SchemaGraph.from_manifest(dataset.schema)
    candidates = enumerate_signal_candidates(dataset, graph, max_hops=MAX_HOPS)
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_CANDIDATE_COUNT} standard RelAmazon candidates, "
            f"found {len(candidates)}; check the source schema and candidate-generation "
            "settings"
        )
    tasks = tuple(_materialize_tasks(candidates))

    candidate_rows = [candidate.to_dict(graph) for candidate in candidates]
    candidate_fingerprint = hashlib.sha256(
        "\n".join(sorted(candidate.id for candidate in candidates)).encode("utf-8")
    ).hexdigest()
    (root / "candidate_catalog.json").write_text(
        json.dumps(candidate_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "tasks.jsonl").open("w", encoding="utf-8") as output:
        for task in tasks:
            output.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "benchmark_version": BENCHMARK_VERSION,
                "description": (
                    "Controlled synthetic condition-to-signal evaluation using the "
                    "real RelAmazon schema and standard 39-candidate pool."
                ),
                "task_count": len(tasks),
                "candidate_count": len(candidates),
                "max_hops": MAX_HOPS,
                "candidate_id_sha256": candidate_fingerprint,
                "schema_name": dataset.schema.name,
                "schema_tables": {
                    table: list(dataset.get_column_names(table))
                    for table in dataset.get_tables()
                },
                "data_note": (
                    "The snapshot contains the source manifest and empty Parquet "
                    "tables with matching schemas only; no source records are copied "
                    "or used for scoring."
                ),
                "oracle_note": (
                    "Grades are author-defined synthetic qrels. They are neither "
                    "model-generated nor human-validated relevance judgments."
                ),
                "reasoning_types": sorted({task.reasoning_type for task in tasks}),
                "grade_meaning": {
                    "0": "irrelevant or distractor",
                    "1": "weak supporting evidence",
                    "2": "useful supporting evidence",
                    "3": "central signal specified by construction",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return tasks


def _write_schema_only_snapshot(source_root: Path, output_root: Path) -> None:
    """Copy the real manifest and Parquet schemas, but no dataset rows."""
    manifest = source_root / "manifest.yaml"
    if not manifest.is_file():
        raise FileNotFoundError(f"RelAmazon manifest not found: {manifest}")
    source = LocalRelBenchDataset(source_root)
    shutil.copyfile(manifest, output_root / "manifest.yaml")
    for table in source.get_tables():
        source_path = source_root / f"{table}.parquet"
        if not source_path.is_file():
            raise FileNotFoundError(f"RelAmazon table schema not found: {source_path}")
        schema = pq.read_schema(source_path)
        empty_table = pa.Table.from_batches([], schema=schema)
        pq.write_table(empty_table, output_root / f"{table}.parquet")


def load_tasks(root: str | Path) -> tuple[ControlledSignalTask, ...]:
    path = Path(root) / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"controlled benchmark tasks are missing: {path}")
    tasks = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            task = ControlledSignalTask(
                id=str(raw["id"]),
                condition=str(raw["condition"]),
                source_table=str(raw["source_table"]),
                reasoning_type=str(raw["reasoning_type"]),
                oracle_rationale=str(raw["oracle_rationale"]),
                signal_grades={
                    str(key): int(value) for key, value in raw["signal_grades"].items()
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid task record") from error
        if not all(
            (
                task.id,
                task.condition,
                task.source_table,
                task.reasoning_type,
                task.oracle_rationale,
            )
        ):
            raise ValueError(f"{path}:{line_number}: task fields must be non-empty")
        if any(grade not in (0, 1, 2, 3) for grade in task.signal_grades.values()):
            raise ValueError(f"{path}:{line_number}: grades must be in the range 0..3")
        tasks.append(task)
    if not tasks:
        raise ValueError(f"controlled benchmark contains no tasks: {path}")
    return tuple(tasks)


def _materialize_tasks(
    candidates: Sequence[SignalCandidate],
) -> Iterable[ControlledSignalTask]:
    by_key = {_signal_key(candidate): candidate.id for candidate in candidates}
    for specification in _TASK_SPECS:
        missing = set(specification["grades"]) - set(by_key)
        if missing:
            raise AssertionError(
                f"task {specification['id']} refers to unavailable real-schema "
                f"signal(s): {sorted(missing)}"
            )
        grades = {
            candidate.id: int(specification["grades"].get(_signal_key(candidate), 0))
            for candidate in candidates
        }
        if not any(grade == 3 for grade in grades.values()):
            raise AssertionError(f"task {specification['id']} has no central signal")
        yield ControlledSignalTask(
            id=str(specification["id"]),
            condition=str(specification["condition"]),
            source_table=str(specification["source_table"]),
            reasoning_type=str(specification["reasoning_type"]),
            oracle_rationale=str(specification["oracle_rationale"]),
            signal_grades=grades,
        )


def _signal_key(candidate: SignalCandidate) -> str:
    path = ">".join(candidate.path)
    if candidate.kind == "relationship":
        return f"relationship:{path}"
    assert candidate.attribute_table is not None and candidate.attribute_column is not None
    return f"attribute:{path}:{candidate.attribute_table}.{candidate.attribute_column}"


# Keys use the exact path/attribute descriptions exposed by the shared candidate
# generator. All referenced fields exist in the real RelAmazon schema.
_TASK_SPECS = (
    {
        "id": "product-type-and-price",
        "condition": "Find products in a similar category and price range.",
        "source_table": "product",
        "reasoning_type": "direct-multiple-signals",
        "oracle_rationale": "Category and price are both explicitly requested.",
        "grades": {
            "attribute:product:product.category": 3,
            "attribute:product:product.price": 3,
        },
    },
    {
        "id": "product-category-under-budget",
        "condition": "Find products in the same broad category that cost less than a budget.",
        "source_table": "product",
        "reasoning_type": "and-filter",
        "oracle_rationale": "The condition requires both product category and price.",
        "grades": {
            "attribute:product:product.category": 3,
            "attribute:product:product.price": 3,
        },
    },
    {
        "id": "product-category-or-description",
        "condition": "Find products in a similar category or with similar descriptions.",
        "source_table": "product",
        "reasoning_type": "or-filter",
        "oracle_rationale": "Either category or description can satisfy the stated alternative.",
        "grades": {
            "attribute:product:product.category": 3,
            "attribute:product:product.description": 3,
        },
    },
    {
        "id": "product-brand-and-category",
        "condition": "Find products from a similar brand and product category.",
        "source_table": "product",
        "reasoning_type": "and-filter",
        "oracle_rationale": "Brand and category are both required comparison signals.",
        "grades": {
            "attribute:product:product.brand": 3,
            "attribute:product:product.category": 3,
        },
    },
    {
        "id": "product-customer-overlap",
        "condition": "Find products that attract many of the same customers.",
        "source_table": "product",
        "reasoning_type": "two-hop-relationship",
        "oracle_rationale": "The product-review-customer path expresses shared customers.",
        "grades": {
            "relationship:product>review>customer": 3,
            "attribute:product>review:review.rating": 1,
        },
    },
    {
        "id": "product-recent-satisfaction",
        "condition": "Find products with similar customer satisfaction in recent reviews.",
        "source_table": "product",
        "reasoning_type": "temporal-one-hop",
        "oracle_rationale": "Rating captures satisfaction and review time captures recency.",
        "grades": {
            "attribute:product>review:review.rating": 3,
            "attribute:product>review:review.review_time": 2,
        },
    },
    {
        "id": "product-verified-feedback",
        "condition": "Compare products using feedback from verified buyers, emphasizing ratings.",
        "source_table": "product",
        "reasoning_type": "one-hop-multiple-signals",
        "oracle_rationale": "Verified status filters reviews and rating expresses satisfaction.",
        "grades": {
            "attribute:product>review:review.verified": 3,
            "attribute:product>review:review.rating": 3,
        },
    },
    {
        "id": "product-positive-feedback-volume",
        "condition": "Find products with consistently positive feedback across their reviews.",
        "source_table": "product",
        "reasoning_type": "aggregation-and-relationship",
        "oracle_rationale": "Ratings are central; the product-review path identifies the review set.",
        "grades": {
            "attribute:product>review:review.rating": 3,
            "relationship:product>review": 2,
        },
    },
    {
        "id": "customer-rating-taste",
        "condition": "Find customers with similar tastes based on the ratings they give.",
        "source_table": "customer",
        "reasoning_type": "one-hop-attribute",
        "oracle_rationale": "Review ratings are the direct evidence of customer preferences.",
        "grades": {
            "attribute:customer>review:review.rating": 3,
            "relationship:customer>review": 1,
        },
    },
    {
        "id": "customer-opinion-and-rating",
        "condition": "Find customers with similar opinions and rating patterns in their reviews.",
        "source_table": "customer",
        "reasoning_type": "one-hop-multiple-signals",
        "oracle_rationale": "Review text and rating are both explicitly relevant.",
        "grades": {
            "attribute:customer>review:review.review_text": 3,
            "attribute:customer>review:review.rating": 3,
        },
    },
    {
        "id": "customer-brand-preferences",
        "condition": "Find customers who tend to review products from similar brands.",
        "source_table": "customer",
        "reasoning_type": "two-hop-attribute",
        "oracle_rationale": "Product brand is reached through customer-review-product.",
        "grades": {
            "attribute:customer>review>product:product.brand": 3,
            "attribute:customer>review:review.rating": 1,
        },
    },
    {
        "id": "customer-category-or-brand",
        "condition": "Find customers whose reviewed products have similar categories or brands.",
        "source_table": "customer",
        "reasoning_type": "two-hop-or-filter",
        "oracle_rationale": "Category and brand are alternative product-side signals.",
        "grades": {
            "attribute:customer>review>product:product.category": 3,
            "attribute:customer>review>product:product.brand": 3,
        },
    },
    {
        "id": "customer-recent-activity",
        "condition": "Find customers with similar recent reviewing activity.",
        "source_table": "customer",
        "reasoning_type": "temporal-one-hop-relationship",
        "oracle_rationale": "Review time and the customer-review path identify recent activity.",
        "grades": {
            "attribute:customer>review:review.review_time": 3,
            "relationship:customer>review": 2,
        },
    },
    {
        "id": "customer-lower-price-ratings",
        "condition": "Find customers who give similar ratings to lower-priced products.",
        "source_table": "customer",
        "reasoning_type": "two-hop-and-one-hop-combination",
        "oracle_rationale": "Product price scopes the products and review rating measures feedback.",
        "grades": {
            "attribute:customer>review>product:product.price": 3,
            "attribute:customer>review:review.rating": 3,
        },
    },
    {
        "id": "product-customer-behavior",
        "condition": "Find products that appeal to similar customers based on their reviewing behavior.",
        "source_table": "product",
        "reasoning_type": "two-hop-relationship",
        "oracle_rationale": "The product-review-customer path represents shared reviewer behavior.",
        "grades": {
            "relationship:product>review>customer": 3,
            "attribute:product>review:review.rating": 1,
        },
    },
    {
        "id": "product-review-text-and-category",
        "condition": "Find products with similar descriptions and customer-written review text.",
        "source_table": "product",
        "reasoning_type": "direct-and-related-text",
        "oracle_rationale": "Product descriptions and review text are the two requested text signals.",
        "grades": {
            "attribute:product:product.description": 3,
            "attribute:product>review:review.review_text": 3,
        },
    },
    {
        "id": "product-review-summary-and-rating",
        "condition": "Compare products using review summaries and their ratings.",
        "source_table": "product",
        "reasoning_type": "one-hop-multiple-signals",
        "oracle_rationale": "Review summary and rating are explicitly named evidence.",
        "grades": {
            "attribute:product>review:review.summary": 3,
            "attribute:product>review:review.rating": 3,
        },
    },
    {
        "id": "product-review-activity",
        "condition": "Find products with substantial review activity and positive typical ratings.",
        "source_table": "product",
        "reasoning_type": "aggregation-proxy",
        "oracle_rationale": "The relationship locates the review set; rating summarizes its quality.",
        "grades": {
            "relationship:product>review": 3,
            "attribute:product>review:review.rating": 2,
        },
    },
    {
        "id": "product-rating-trend",
        "condition": "Find products whose customer ratings have improved over time.",
        "source_table": "product",
        "reasoning_type": "temporal-trend",
        "oracle_rationale": "A trend requires review timestamps and rating values.",
        "grades": {
            "attribute:product>review:review.review_time": 3,
            "attribute:product>review:review.rating": 3,
        },
    },
    {
        "id": "product-category-or-budget",
        "condition": "Find products that are either in a target category or below a price limit.",
        "source_table": "product",
        "reasoning_type": "or-filter",
        "oracle_rationale": "Either direct category or direct price can satisfy the condition.",
        "grades": {
            "attribute:product:product.category": 3,
            "attribute:product:product.price": 3,
        },
    },
    {
        "id": "verified-recent-ratings",
        "condition": "Compare recent ratings from verified purchasers across products.",
        "source_table": "product",
        "reasoning_type": "temporal-and-filter-combination",
        "oracle_rationale": "Verification, review date, and rating are all required.",
        "grades": {
            "attribute:product>review:review.verified": 3,
            "attribute:product>review:review.review_time": 3,
            "attribute:product>review:review.rating": 3,
        },
    },
    {
        "id": "customer-satisfaction-by-category",
        "condition": "Find customers with similar satisfaction patterns across product categories.",
        "source_table": "customer",
        "reasoning_type": "two-hop-grouped-pattern",
        "oracle_rationale": "Rating expresses satisfaction and category is the grouping dimension.",
        "grades": {
            "attribute:customer>review:review.rating": 3,
            "attribute:customer>review>product:product.category": 3,
        },
    },
    {
        "id": "customer-verified-review-pattern",
        "condition": "Find customers with similar rating patterns in reviews of verified purchases.",
        "source_table": "customer",
        "reasoning_type": "one-hop-filter-combination",
        "oracle_rationale": "Rating is the pattern and verified status filters the reviews.",
        "grades": {
            "attribute:customer>review:review.rating": 3,
            "attribute:customer>review:review.verified": 3,
        },
    },
    {
        "id": "customer-price-or-description",
        "condition": "Find customers whose reviewed products are similar in price or description.",
        "source_table": "customer",
        "reasoning_type": "two-hop-or-multiple-signals",
        "oracle_rationale": "Price and description are alternative product attributes.",
        "grades": {
            "attribute:customer>review>product:product.price": 3,
            "attribute:customer>review>product:product.description": 3,
        },
    },
    {
        "id": "customer-name-similarity",
        "condition": "Find customer records with similar names.",
        "source_table": "customer",
        "reasoning_type": "direct-attribute",
        "oracle_rationale": "Customer name is the direct attribute specified by the condition.",
        "grades": {"attribute:customer:customer.customer_name": 3},
    },
)

