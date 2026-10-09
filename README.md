# Similarity_benchmark

A benchmark for similarity over relational databases, built on
[RelBench](https://relbench.stanford.edu/). The initial development dataset is
**rel-amazon** (customers, products and time-stamped reviews).

The project currently covers the data foundation: reading a dataset's schema,
representing it as a graph, and building a small, validated local subset to
develop against. It also has a first, schema-only signal-discovery baseline.
Similarity computation and benchmark generation are not implemented yet.

## Layout

```
data/
  manifests/<dataset>/manifest.yaml   source schema manifest (tracked input)
  dev/<dataset>/                      generated development subset (git-ignored)
    review.parquet, product.parquet, customer.parquet, manifest.yaml
src/
  datasets/
    relbench_manifest.py      parse a manifest into dataclasses
    relational_dataset.py     storage-independent relational dataset interface
    local_relbench_dataset.py lazy access to a local Parquet subset
    dev_subset.py             DevSubsetConfig: subset size, sampling mode, paths
    dev_subset_builder.py     build a subset from the remote parquet files
    dev_subset_validation.py  integrity checks and statistics for a subset
  schema/
    schema_graph.py           SchemaGraph: tables and foreign keys as a graph
  signals/
    candidates.py             direct attributes and relational-path candidates
    scorers.py                MiniLM cross-encoder and Decider decision backends
scripts/
  inspect_manifest.py         print a parsed manifest
  build_dev_subset.py         build data/dev/<dataset>/
  validate_dev_subset.py      validate a subset and print statistics
  discover_signals.py         enumerate and score schema signals for conditions
tests/                        unittest suites, all offline
```

## Setup

Python 3.10+ with `duckdb`, `pyarrow` and `pyyaml` (developed with Python 3.14,
DuckDB 1.5, PyArrow 25, PyYAML 6), for example in a conda environment:

```bash
conda create -n simbench python=3.14
conda activate simbench
pip install duckdb pyarrow pyyaml
```

All commands below run from the repository root.

## Schema: manifest and graph

A RelBench manifest lists each table's primary key, time column and foreign
keys, plus the validation/test split timestamps. It does **not** list a table's
other columns; those exist only in the parquet files.

```bash
python scripts/inspect_manifest.py                      # rel-amazon by default
python scripts/inspect_manifest.py path/to/manifest.yaml --json
```

The manifest is the single source of truth for table names and keys; nothing
else hard-codes them. `SchemaGraph` builds on the parsed manifest:

```python
from src.datasets.relbench_manifest import load_manifest
from src.schema.schema_graph import SchemaGraph

graph = SchemaGraph.from_manifest(load_manifest("data/manifests/rel-amazon/manifest.yaml"))
graph.get_tables()                       # TableSchema objects
graph.get_edges()                        # Edge(source_table, source_column, target_table, target_column)
graph.get_neighbors("review")            # ["customer", "product"]
graph.get_relationship("product", "review")  # review.product_id -> product.product_id, either direction
graph.find_paths("product", max_hops=2)  # [["product", "review"], ["product", "review", "customer"]]
```

`get_relationship` returns `None` for unrelated tables and raises
`SchemaGraphError` for an unknown table or an ambiguous pair. `find_paths`
returns simple, deterministic paths that follow foreign keys in both directions.
The shared `ForeignKey` object preserves the complete relationship: source
column, target table and resolved target primary-key column. For example,
`review.product_id -> product.product_id`.

## Relational dataset API

`RelationalDataset` is the storage-independent interface used by later
pipeline stages. It exposes table access and manifest-derived metadata without
requiring callers to know whether data comes from local Parquet, an in-memory
table or another backend:

```python
dataset.get_tables()
dataset.has_table("product")
dataset.get_table("product")
dataset.get_primary_key("product")
dataset.get_foreign_keys()
dataset.get_time_column("review")
dataset.get_entity("product", product_id)
dataset.get_rows("review", "product_id", product_id)
```

`LocalRelBenchDataset` implements this interface for a generated local subset.
It parses `manifest.yaml` during construction but does not load any Parquet
table until that table is requested. Loaded tables are cached as PyArrow
tables.

```python
from src.datasets.local_relbench_dataset import LocalRelBenchDataset

dataset = LocalRelBenchDataset()  # data/dev/rel-amazon/ by default

dataset.get_tables()
# ('review', 'product', 'customer')

dataset.get_primary_key("product")
# 'product_id'

dataset.get_time_column("review")
# 'review_time'

product_id = dataset.get_table("product")["product_id"][0].as_py()
product = dataset.get_entity("product", product_id)  # one-row pa.Table
reviews = dataset.get_rows("review", "product_id", product_id)
```

`get_entity()` uses the primary key declared in the manifest and returns a
one-row table. `get_rows()` returns all rows equal to the supplied value.
Unknown tables, unknown columns and missing entities raise clear errors. These
methods intentionally do not perform joins or path traversal; relational
traversal remains the responsibility of `SchemaGraph`.

## Development subset

The full rel-amazon `review.parquet` has about 20.9 million rows (7.2 GB) in 20
time-sorted row groups. The builder never downloads it. DuckDB queries the
remote file on Hugging Face directly and reads only:

- a spread of evenly spaced row groups (10 by default), which covers 2008-2018;
- the `product_id` (and, in overlap-aware mode, `customer_id`) column to pick
  products, then whole rows only for the sampled reviews.

Each build streams roughly 3.6 GB and takes one to two minutes. Products and
customers are then fetched only for the keys the kept reviews reference, so
every foreign key in the subset resolves.

### Sampling modes

Both modes pick products with at least `min_reviews_per_product` reviews (counted
in the sampled row groups), select up to `num_products`, and keep at most
`max_reviews_per_product` reviews each. Both are deterministic for a given
config.

| Mode | Picks products | Use |
|---|---|---|
| `entity_centered` (default) | pseudo-randomly, by an md5 hash of the product id and `seed` | a representative sample |
| `overlap_aware` | by growing a set of products that share customers | developing signals such as `Product -> Review -> customer_id` |

Overlap-aware sampling starts from the eligible product with the most customers
who also reviewed another eligible product. It then repeatedly adds the product
sharing the most distinct customers with those already chosen. When nothing
left shares a customer, it starts again from the next-best seed. When capping a
product's reviews, reviews by shared customers are kept first.

Overlap-aware sampling exists only to make a useful development subset. It is
not part of the benchmark methodology and should not be assumed for evaluation
datasets.

### Configuration

| Setting | Default | CLI flag |
|---|---|---|
| `dataset` | `rel-amazon` | `--dataset` |
| `num_products` | 500 | `--num-products` |
| `min_reviews_per_product` | 20 | `--min-reviews-per-product` |
| `max_reviews_per_product` | 100 | `--max-reviews-per-product` |
| `sampling_mode` | `entity_centered` | `--sampling-mode` |
| `seed` | 0 | `--seed` |
| `review_row_groups` | 10 | `--review-row-groups` |
| `output_dir` | `data/dev/<dataset>` | `--output-dir` |

The subset holds at most `num_products × max_reviews_per_product` reviews
(`DevSubsetConfig.review_rows`, 50,000 by default). Fewer row groups means a
smaller download but a narrower pool to sample from.

### Build and validate

```bash
python scripts/build_dev_subset.py                                # entity-centered
python scripts/build_dev_subset.py --sampling-mode overlap_aware  # overlap-aware
python scripts/validate_dev_subset.py
```

A build overwrites `data/dev/<dataset>/`. The validator exits with status 1 if
any check fails. It checks that the manifest and every table exist and are
non-empty, that the review count is within the configured bound, that primary
keys are unique, and that every foreign key resolves. It also prints:

- row counts and unique products and customers;
- min/median/mean/max reviews per product and per customer;
- the `review_time` range;
- product overlap through shared customers: products sharing a customer,
  the share of product pairs with nonzero overlap, shared customers per
  overlapping pair, customers linking two or more products, and the median
  nonzero Jaccard similarity of customer sets.

If the subset was built with non-default sizes, pass the same
`--num-products` and `--max-reviews-per-product` to the validator.

### Current subsets (rel-amazon, default sizes)

`data/dev/rel-amazon/` currently holds the overlap-aware subset. Both modes pass
every validation check.

| | Entity-centered | Overlap-aware |
|---|---|---|
| Reviews / products / customers | 25,632 / 500 / 24,361 | 50,000 / 500 / 43,948 |
| Reviews per product (min / median / max) | 20 / 41 / 100 | 100 / 100 / 100 |
| Products sharing a customer | 83.8% | 100% |
| Product pairs with nonzero overlap | 1.02% | 5.07% |
| Shared customers per overlapping pair (median / mean / max) | 1 / 1.20 / 16 | 1 / 1.14 / 15 |
| Customers linking 2+ products | 987 | 5,132 |
| Median nonzero Jaccard | 0.0083 | 0.0050 |

Customers are sparse in both: about 1.1 reviews per customer. Overlap-aware
sampling connects many more products, but most overlapping pairs still share a
single customer.

## Schema signal discovery

The first signal-discovery baseline is deliberately independent of data values:
it enumerates all ordinary columns reachable from each anchor table through
simple foreign-key paths. `--hop-limit` bounds the maximum number of joins;
the default is two. Primary keys and foreign-key identifiers are excluded by
default, while temporal columns remain candidates. Path-only relationship
candidates are included so a scorer can select a relationship even when no
single downstream column is decisive.

The same candidate list can be compared with local scoring approaches:

- `minilm`: a conventional Hugging Face MiniLM cross-encoder. Its sigmoid score
  is a ranking value, not a calibrated relevance probability.
- `qwen3-reranker`: an instruction-aware Qwen3 pairwise reranker. It scores
  each intent/signal pair independently and is also not a calibrated
  probability of relevance.
- `decider`: a locally loaded Decider/Jev-style model. It selects among all
  supplied signals and returns one probability distribution across that exact
  candidate set. It is limited to 255 candidates.

Install a matching backend before running it, for example:

```bash
pip install torch transformers                         # MiniLM
pip install "transformers>=4.51" accelerate            # Qwen3 reranker (plus torch)
pip install decider-ai flash-linear-attention           # Decider on CUDA
```

Inspect candidates before loading a model:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions3.jsonl \
  --source-table product \
  --hop-limit 2 \
  --list-only \
  --output results/rel-amazon/schema-candidates.jsonl
```

Run the two comparable local scoring baselines:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions3.jsonl \
  --hop-limit 2 --scorer minilm \
  --output results/rel-amazon/minilm-signals.jsonl

python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions3.jsonl \
  --hop-limit 2 --scorer decider \
  --model-name Mapika/decider-4b \
  --output results/rel-amazon/decider-signals.jsonl

python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions3.jsonl \
  --hop-limit 2 --scorer qwen3-reranker \
  --model-name Qwen/Qwen3-Reranker-4B \
  --batch-size 8 --device cuda \
  --output results/rel-amazon/qwen3-reranker-signals.jsonl

```

Each JSONL result retains the input condition, candidate-generation settings,
full candidate count and the top-ranked candidates. Treat the returned scores
as model outputs to evaluate against manual relevance judgements, not as ground
truth relevance labels.

## Tests

```bash
python -m unittest discover tests
```

The tests run offline, using the rel-amazon manifest and small synthetic
Parquet files. They cover manifest metadata (including resolved foreign-key
target columns), `SchemaGraph`, the generic/local dataset APIs and lazy loading,
the subset config, both sampling modes and the validator. Set `VERBOSE = 1`
(default) or `0` at the top of a test file to show or hide its diagnostic
printout.
