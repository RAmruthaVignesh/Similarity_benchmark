# Similarity_benchmark

A benchmark for similarity over relational databases, built on
[RelBench](https://relbench.stanford.edu/). The initial development dataset is
**rel-amazon** (customers, products and time-stamped reviews).

The project covers schema-based signal discovery and conditioned similarity
over relational entities. It includes discovery rankings evaluated against
human signal labels, signal-specific similarity measures, external baseline
comparisons, and exploratory end-to-end tools. The similarity-scoring and
ranking evaluation work is experimental and is being developed alongside its
analysis notebooks.

## Layout

```
data/
  manifests/<dataset>/manifest.yaml   source schema manifest (tracked input)
  dev/<dataset>/                      generated development subset (git-ignored)
    review.parquet, product.parquet, customer.parquet, manifest.yaml
conditions/
  <dataset>/conditions.jsonl          static conditions (tracked input)
annotations/
  <dataset>/human/<annotator>.jsonl   reviewable human relevance judgments
results/
  <dataset>/discovery/                ranked signal outputs
docs/                                 protocols, exact prompts, and design notes
notebooks/                            interactive result analysis
src/
  llm_backends.py             shared local and OCI model setup
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
    scorers.py                MiniLM and Strands Decider backend implementations
    llm_discovery.py          pointwise generative-LLM backend and prompts
scripts/
  inspect_manifest.py         print a parsed manifest
  build_dev_subset.py         build data/dev/<dataset>/
  validate_dev_subset.py      validate a subset and print statistics
  discover_signals.py         enumerate and score schema signals for conditions
tests/                        unittest suites, all offline
```

`data/` contains source inputs and local development subsets; `annotations/`
contains human relevance judgments; and `results/` contains outputs derived
from discovery and similarity experiments. The Markdown files in `docs/`
describe the current methods, prompts, and result formats.

The RelAmazon condition file contains 15 conditions. Existing IDs were made
unique when the three sets were merged: former v1 IDs 001–005 remain 001–005,
former v2 IDs 001–005 are now 006–010, and former v3 IDs 001–005 are now
011–015. Human judgments, oracle roots, and saved discovery runs
use the same IDs. Runs that evaluated only some conditions keep that coverage;
for example, the saved Astra run covers four conditions.

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

Signal discovery has two independent settings: a **method** controls how the
catalog is explored, and a validated **model** profile provides ranking or
action-selection.

Methods are:

- `catalog-rank`: score every candidate in the fixed catalog and rank it.
- `graph-search`: select legal graph actions, then rerank only the signals
  discovered by the search.
- `graph-search-oracle-root`: the graph-search diagnostic with the initial
  table fixed by an oracle-root map.

Supported model profiles are:

- `minilm`: a conventional Hugging Face MiniLM cross-encoder. Its sigmoid score
  is a ranking value, not a calibrated relevance probability.
- `strands-decider`: the Gemma-4 Strands typed-choice model
  `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`. It has the same
  categorical-probability output contract and is limited to 255 candidates.
- `random`: a seeded random ordering over the same enumerated candidate list.
  Use it as a chance baseline; it does not load a model.

`--method graph-search --model strands-decider` first has the Strands
Decider explore
legal graph actions (select a table, emit an attribute/path, traverse one
foreign key, or stop), then performs a final multi-choice ranking over only the
discovered shortlist. The final ranking prompt contains the intent and a
summary of the graph-search decisions—not the full schema. The complete
state/action trace is saved with the result. Final choice probabilities are
conditional on the discovered shortlist; undiscovered signals cannot be
recovered by the final ranking.

`graph-search-oracle-root` uses the same search and reranking process,
but fixes the initial table from a condition-to-table JSON mapping. For the
merged RelAmazon conditions file, `conditions/rel-amazon/oracle-root-map.json`
maps each condition to the entity type it explicitly asks to retrieve (product,
review, or customer). This is a **target-entity proxy**, not a human-validated
gold root: useful signals can start from other entities, so interpret this as a
root-choice ablation rather than the definitive best possible root. The root
choice consumes the same one-call budget as model-selected root selection.

Install a matching backend before running it, for example:

```bash
pip install torch transformers                         # MiniLM
# Strands Gemma-4 Decider (Gemma-4 support is currently on the Git main branch)
pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"
```

Inspect candidates before loading a model:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --source-table product \
  --hop-limit 2 \
  --list-only \
  --output results/rel-amazon/discovery/schema-candidates.jsonl
```

### Path-aware and feature-collapsed catalog views

`path-aware` is the default 39-signal RelAmazon catalog: a signal contains its
endpoint feature *and* its source/path; it also includes relationship-path
signals. `feature-collapsed` is a separate, smaller task with exactly one
direct candidate per physical table/column and no relationship/path candidates.
It answers whether a method identifies the useful feature at all, independently
of its chosen provenance path. The collapsed view reuses the direct-path qrel
for each feature; it does not require a second annotation pass.

Run either view with `--catalog-view path-aware` or
`--catalog-view feature-collapsed`. For convenience, `--catalog-view both`
runs the method twice and appends `-path-aware` and `-feature-collapsed` to the
given output filename:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method catalog-rank --model minilm --catalog-view both \
  --output results/rel-amazon/discovery/minilm.jsonl
```

The CLI keeps the **method** independent from the validated **model** profile:

- `catalog-rank` scores every candidate in the selected catalog. It supports
  `random`, `minilm`, `strands-decider`, and each generative-LLM model profile.
- `graph-search` programmatically explores legal schema actions and reranks
  only its discovered shortlist. It supports `strands-decider` and each
  generative-LLM model profile.
- `graph-search-oracle-root` is the same graph-search strategy with an
  externally supplied starting table; use it only as a diagnostic.

Generative profiles such as `muse-glimmer-30b`, `gpt-6-luna`, `gpt-6-sol`,
`gpt-5.6-terra`, and `gpt-6-astra` select the configured local-vLLM or OCI
runner through the same `--model` flag. Raw model identifiers are intentionally
not accepted: add a validated profile in
`src/signals/model_registry.py` for a new compatible checkpoint.

Do not compare nDCG across these two catalogs as if they were the same task.
Compare methods within each view; report both to distinguish feature discovery
from path/provenance discovery.

Run the two comparable local scoring baselines:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method catalog-rank --model minilm \
  --output results/rel-amazon/discovery/minilm-signals.jsonl

# Strands' Gemma-4 typed-choice Decider. --device auto selects CUDA when
# available. The first invocation downloads both the adapter and Gemma base
# weights into the configured Hugging Face cache.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method catalog-rank --model strands-decider \
  --output results/rel-amazon/discovery/strands-decider-e4b-signals.jsonl

# Strands-guided graph discovery followed by reranking from the explored-schema
# view of only the discovered signals.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method graph-search --model strands-decider \
  --max-decisions 12 --branch-width 2 --max-signals 10 \
  --output results/rel-amazon/discovery/strands-decider-e4b-graph-reranked-signals.jsonl

# Target-entity-proxy root ablation for the merged conditions.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method graph-search-oracle-root --model strands-decider \
  --oracle-root-map conditions/rel-amazon/oracle-root-map.json \
  --max-decisions 30 --branch-width 3 --max-signals 10 \
  --output results/rel-amazon/discovery/strands-decider-e4b-graph-oracle-root.jsonl

# Reproducible random ranking baseline (no model weights are loaded).
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --hop-limit 2 --method catalog-rank --model random --seed 42 \
  --output results/rel-amazon/discovery/random-seed42.jsonl

```

Each JSONL result retains the input condition, candidate-generation settings,
full candidate count and the top-ranked candidates. Treat the returned scores
as model outputs to evaluate against manual relevance judgements, not as ground
truth relevance labels.

### Generative-LLM discovery

Generative-LLM model profiles score each condition/signal pair independently with a
generative LLM. By default it returns a **0--10 utility score** and short
evidence note; this is a discovery utility score, not a probability or a human relevance grade. Use `--utility-scale 0-100` to request the earlier
scale. With `--method graph-search`, the same pointwise score evaluates
each legal graph action, retains the top `--branch-width` actions by within-menu
ordinal rank, then reranks only the discovered shortlist. The
`graph-search-oracle-root` strategy is the target-entity-proxy diagnostic.

The local models run through vLLM; GPT-6 Luna uses OCI through WayFlow. Pass `--model-cache-root` to keep Hugging Face weights out
of `/home`, and `--resume` to persist completed conditions as they finish.

```bash
# Full-catalog schema reasoning with a local model.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --dataset-root data/dev/rel-amazon \
  --method catalog-rank --model muse-glimmer-30b \
  --hop-limit 2 --batch-size 4 --max-tokens 256 --max-attempts 2 \
  --model-cache-root /storage/lcostes/model-cache/huggingface \
  --resume --output results/rel-amazon/discovery/llm-muse-schema.jsonl

# LLM-guided graph search and LLM shortlist reranking.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --dataset-root data/dev/rel-amazon \
  --method graph-search --model muse-glimmer-30b \
  --hop-limit 2 --max-decisions 12 --branch-width 2 --max-signals 10 \
  --batch-size 4 --max-tokens 256 \
  --model-cache-root /storage/lcostes/model-cache/huggingface \
  --resume --output results/rel-amazon/discovery/llm-muse-graph.jsonl
```

Evaluate these runs against the human qrels in
`notebooks/analyze_signal_discovery_results.ipynb`. The current discovery
prompts are documented in [signal-discovery-prompts.md](docs/signal-discovery-prompts.md).

## Signal-specific entity similarity (proposal §3.2)

`scripts/score_entity_similarity.py` is the first deterministic implementation
of §3.2. It takes signals already selected in §3.1, materializes each signal
for every entity in its source table, and computes transparent entity-pair
scores. It does **not** select signals, use relevance labels, learn weights, or
turn scores into final similarity grades yet.

The automatic measure selection is type- and path-aware while remaining
inspectable:

- direct numeric values use empirical-percentile distance; multi-valued numeric
  paths use percentile-normalized one-dimensional Wasserstein similarity. This
  is robust to numeric outliers such as a few unusually expensive products;
- dates and timestamps use the same scalar/distribution distinction on their
  empirical temporal percentiles;
- list-valued categorical fields are flattened, HTML-normalized, and compared
  by IDF-weighted Jaccard. Common labels therefore contribute less than a
  discriminative label; scalar categoricals use equality. Entity-name fields
  are treated as categorical identifiers rather than semantic text;
- a multi-valued boolean field is compared by its rate (for example, the share
  of verified reviews), rather than by the uninformative set `{True, False}`;
- text-like fields (`description`, `title`, `summary`, or `*_text`) use
  contextual E5 embeddings and cosine similarity, including for path-derived
  text collections;
- multi-row evidence reached through a path receives a visible support-based
  confidence factor when signals are combined. A collection with one related
  row cannot dominate one supported by many rows;
- relationship paths ending at identified entities use neighborhood Jaccard.
  A path ending in rows with no primary key instead uses degree similarity,
  rather than pretending anonymous rows form a reusable neighbour set.

Four explicit alternatives are also available through `--measure-variant` or
per-signal `--signal-measure SIGNAL_ID=VARIANT` overrides:

- `jensen-shannon` compares categorical/boolean frequencies, endpoint sets, or
  quantile-binned numeric and temporal path distributions. It reports
  `1 - sqrt(JS)` using base-2 divergence, so scores stay in `[0, 1]`. For
  relationship signals, the endpoint table must have a primary key; anonymous
  row positions are not meaningful shared categories.
- `hellinger` compares the same kinds of distributions and reports
  `1 - Hellinger distance`, also in `[0, 1]`, with the same stable-endpoint-ID
  requirement for relationship signals.
- `wl-subtree-kernel` is for relationship signals. It builds a rooted,
  table- and foreign-key-typed ego-graph restricted to the signal path, applies
  Weisfeiler–Lehman neighborhood relabeling, then compares the resulting
  subtree-count vectors with cosine similarity.
- `diffusion-distance` is for relationship signals. It projects shared
  endpoint entities into a weighted graph over the source entities, downweights
  popular endpoints, adds a 0.2 restart mass, and compares one-step random-walk
  profiles by normalized Euclidean distance. The endpoint table must have a
  primary key; entities with no shared-endpoint neighbors have no diffusion
  score.

These variants are explicit experiments; `auto` remains the default. The
distribution alternatives require discrete values or multi-valued numeric or
temporal paths. The graph variants require relationship signals. The
diffusion projection has an edge-count guard to avoid unexpectedly large
materializations.

For example, score products relative to product `197` using the top discovery
signal for `condition-001` that starts from `product`. When a discovery file is
used, `--top-signals` counts only signals rooted at `--entity-table`; it never
averages paths rooted at another entity type. The output has one JSONL record
per query and keeps the combined score plus every individual signal/measure
component for each candidate entity.

```bash
python scripts/score_entity_similarity.py \
  --conditions conditions/rel-amazon/conditions.jsonl \
  --condition-id condition-001 \
  --dataset-root data/dev/rel-amazon \
  --entity-table product \
  --discovery-results results/rel-amazon/discovery/catalog-views/catalog-rank-gpt-6-luna-utility-0-10-high-path-aware.jsonl \
  --top-signals 1 \
  --measure-variant auto \
  --query-id 197 \
  --top-k 20 \
  --model-cache-root /storage/lcostes/model-cache/huggingface/hub \
  --output results/rel-amazon/similarity/condition-001-product-197.jsonl
```

To compare a relationship signal with the WL subtree kernel, replace the
discovery source with an explicit relationship signal and set
`--measure-variant wl-subtree-kernel`. Use `--measure-variant diffusion-distance`
for the one-step diffusion score. When selected signals need different
measures, keep `--measure-variant auto` and override individual IDs, for example
`--signal-measure 'SIGNAL_ID=hellinger'`.

Text values are encoded with the contextual `intfloat/e5-small-v2` encoder
through vLLM's batched pooling runner. For a path that yields many texts per
entity (for example a product's 100 reviews), the current implementation
averages individual text embeddings; that collection representation can become
too similar across entities and should be treated as a baseline, not the final
review-profile measure. No model is downloaded automatically by the similarity
layer.

Text vectors are cached persistently by model snapshot and exact E5 input, so
the same text value is embedded only once across signals and path variants. To
warm the cache for the local RelAmazon subset, run this once from the SimBench
environment; rerunning it is resumable and skips cached vectors:

```bash
export SIMBENCH_TMPDIR=/storage/lcostes/tmp
export TMPDIR="$SIMBENCH_TMPDIR"
export LD_LIBRARY_PATH="/storage/lcostes/ray-share/venvs/simbench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/precompute_text_embeddings.py \
  --dataset-root data/dev/rel-amazon \
  --model-cache-root /storage/lcostes/model-cache/huggingface/hub \
  --embedding-cache /storage/lcostes/simbench-cache/text-embeddings.sqlite3
```

The explorer notebooks and `score_entity_similarity.py` use this cache by
default. Override its path with `SIMBENCH_TEXT_EMBEDDING_CACHE` if needed.

Use explicit selected candidate IDs instead of a discovery result when you want
to test a hand-authored signal set. Several signals are combined by a weighted
mean of their available component scores; specify a non-default weight with
`--signal-weight 'SIGNAL_ID=WEIGHT'`. The next §3.2 stages are to validate
which measures and combinations represent each condition, then learn or
otherwise calibrate the combination and convert the resulting scores into
similarity levels.

For interactive inspection without relying on notebook widgets, start the
local browser explorer:

```bash
conda run -p /storage/lcostes/ray-share/venvs/simbench \
  python scripts/explore_similarity_ui.py
```

The server tries port `8765` and automatically searches the next 49 ports if it
is occupied; its terminal prints the port it actually selected. On the remote
compute host, use VS Code's **Ports** view to forward that printed port, then
open the forwarded address. You can set the starting port with `--port` or the
search range with `--port-attempts`. The UI
lets you choose a condition and discovery run, restricts selection to signals
rooted at the retrieved entity type, changes weights, and displays component
scores plus raw nearest records. It runs locally and makes no external model or
API calls.

The notebook version is interactive too and uses `ipywidgets`; select the base
Python kernel where that package is installed, then rerun its cells.

[`notebooks/demo_end_to_end_similarity.ipynb`](notebooks/demo_end_to_end_similarity.ipynb)
provides a separate end-to-end POC: enter free-form condition text, run
catalog signal discovery with an OCI model, select compatible signals and
weights, and inspect the resulting nearest-entity ranking. It uses the
SimBench Conda environment for discovery and the notebook's base kernel for
the interactive controls. Its current OCI compartment is configured directly
in the notebook.

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
