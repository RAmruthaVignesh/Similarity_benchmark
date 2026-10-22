# Similarity_benchmark

A benchmark for similarity over relational databases, built on
[RelBench](https://relbench.stanford.edu/). The initial development dataset is
**rel-amazon** (customers, products and time-stamped reviews).

The project covers the data foundation: reading a dataset's schema,
representing it as a graph, and building a small, validated local subset to
develop against. It includes schema-only signal discovery and a controlled
evaluation of that stage. End-to-end user-conditioned embeddings and similarity
evaluation remain under development.

## Controlled signal-discovery evaluation

The repository includes a **controlled synthetic** evaluation for the
signal-discovery stage. It uses the actual RelAmazon manifest and Parquet
schemas, and the same full 39-candidate signal pool used by standard runs.
The builder snapshots only schema metadata into empty Parquet tables; it does
not copy or use source records. Its 25 conditions cover direct attributes,
one- and two-hop paths, relationship evidence, multiple relevant signals,
distractors, AND/OR logic, and temporal reasoning. Each task has explicit
graded qrels and a rationale. The qrels are author-defined synthetic oracles,
not human-validated judgments. This tests whether a ranker follows designed
signal rules; it does not evaluate entity embeddings or replace human
judgments on real conditions.

```bash
python scripts/build_controlled_signal_benchmark.py \
  --schema-root data/dev/rel-amazon \
  --output-root benchmarks/controlled-signal-discovery/rel-amazon-v2

python scripts/evaluate_controlled_signals.py \
  --scorer random \
  --scorer minilm \
  --scorer strands-decider \
  --benchmark-root benchmarks/controlled-signal-discovery/rel-amazon-v2 \
  --output results/rel-amazon/controlled-signal-discovery/v2/summary.json
```

The summary reports macro nDCG@1/3/5, Recall@1/3/5 for central (grade-3)
signals, and MRR of the first central signal. The adjacent `*-tasks.jsonl`
file records the top-ranked signals and per-condition scores for inspection.
The scorer classes and prompt construction are shared with standard signal
discovery; only the condition and synthetic qrels differ.

## Layout

```
data/
  manifests/<dataset>/manifest.yaml   source schema manifest (tracked input)
  dev/<dataset>/                      generated development subset (git-ignored)
    review.parquet, product.parquet, customer.parquet, manifest.yaml
conditions/
  <dataset>/conditions-vN.jsonl       versioned static user intents (tracked input)
annotations/
  <dataset>/human/<annotator>.jsonl   reviewable human relevance judgments
  <dataset>/llm/<prompt-version>/     generated model judgments (git-ignored)
benchmarks/
  controlled-signal-discovery/<dataset>-vN/
                                      generated schema snapshots and oracle qrels
results/
  <dataset>/discovery/                ranked signal outputs
  <dataset>/controlled-signal-discovery/vN/
                                      synthetic-benchmark summaries and task rows
  <dataset>/llm-judge/                judge metrics and throughput profiles
docs/                                 protocols, exact prompts, and design notes
notebooks/                            interactive result analysis
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
    scorers.py                MiniLM and Strands Decider backends
  synthetic/
    controlled_signal_benchmark_v2.py  real-schema controlled qrels
scripts/
  inspect_manifest.py         print a parsed manifest
  build_dev_subset.py         build data/dev/<dataset>/
  validate_dev_subset.py      validate a subset and print statistics
  discover_signals.py         enumerate and score schema signals for conditions
tests/                        unittest suites, all offline
```

`data/` contains source inputs and local development subsets; `benchmarks/`
contains generated benchmark fixtures and oracle labels; `annotations/`
contains relevance judgments; and `results/` contains outputs derived from a
method or evaluation. See
[`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md) for naming and artifact
retention rules.

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
- `strands-decider`: the Gemma-4 Strands typed-choice model
  `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`. It has the same
  categorical-probability output contract and is limited to 255 candidates.
- `random`: a seeded random ordering over the same enumerated candidate list.
  Use it as a chance baseline; it does not load a model.

`--method decider-graph-reranked --scorer strands-decider` first has the
Strands Decider explore
legal graph actions (select a table, emit an attribute/path, traverse one
foreign key, or stop), then performs a final multi-choice ranking over only the
discovered shortlist. The final ranking prompt contains the intent and a
summary of the graph-search decisions—not the full schema. The complete
state/action trace is saved with the result. Final choice probabilities are
conditional on the discovered shortlist; undiscovered signals cannot be
recovered by the final ranking.

`decider-graph-reranked-oracle-root` uses the same search and reranking process,
but fixes the initial table from a condition-to-table JSON mapping. For the
current RelAmazon conditions-v3 file, `conditions/rel-amazon/oracle-root-map-v3.json`
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
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
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
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer minilm --catalog-view both \
  --output results/rel-amazon/discovery/minilm.jsonl
```

Do not compare nDCG across these two catalogs as if they were the same task.
Compare methods within each view; report both to distinguish feature discovery
from path/provenance discovery.

Run the two comparable local scoring baselines:

```bash
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer minilm \
  --output results/rel-amazon/discovery/minilm-signals.jsonl

# Strands' Gemma-4 typed-choice Decider. --device auto selects CUDA when
# available. The first invocation downloads both the adapter and Gemma base
# weights into the configured Hugging Face cache.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer strands-decider \
  --model-name StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --output results/rel-amazon/discovery/strands-decider-e4b-signals.jsonl

# Strands-guided graph discovery followed by reranking from the explored-schema
# view of only the discovered signals.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer strands-decider --method decider-graph-reranked \
  --max-decisions 12 --branch-width 2 --max-signals 10 \
  --model-name StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --output results/rel-amazon/discovery/strands-decider-e4b-graph-reranked-signals.jsonl

# Target-entity-proxy root ablation for conditions-v3.
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer strands-decider \
  --method decider-graph-reranked-oracle-root \
  --oracle-root-map conditions/rel-amazon/oracle-root-map-v3.json \
  --max-decisions 30 --branch-width 3 --max-signals 10 \
  --model-name StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --output results/rel-amazon/discovery/strands-decider-e4b-graph-oracle-root.jsonl

# Reproducible random ranking baseline (no model weights are loaded).
python scripts/discover_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --hop-limit 2 --scorer random --seed 42 \
  --output results/rel-amazon/discovery/random-seed42.jsonl

```

Each JSONL result retains the input condition, candidate-generation settings,
full candidate count and the top-ranked candidates. Treat the returned scores
as model outputs to evaluate against manual relevance judgements, not as ground
truth relevance labels.

## LLM-as-a-judge signal relevance evaluation

`scripts/judge_signals.py` produces **blinded, schema-only, pair-level** LLM
judgments. Each model sees a condition, one candidate signal (including its
complete path), and the schema. It never sees
a discovery method, its score, its rank, or graph-search history. Each run
always evaluates three wording-only prompt variants with the same rubric and
decoding configuration. It writes their resumable raw labels in a `variants/`
subdirectory and writes a consensus JSONL at `--output`. The consensus grade
is the median of the three ordinal grades: this is the majority grade whenever
one exists, and selects the middle grade for a three-way tie. Completed raw
labels are reused only when the condition, candidate, schema/candidate catalog,
prompt, model revision, and sampling configuration have the same cache key.

The local presets are Mistral Small 3.2 (24B), Meta Muse Glimmer 30B, Google
Gemma 4 31B IT, and OpenAI GPT-OSS-120B. All use vLLM's offline batched
engine, JSON-Schema-constrained decoding, and prefix caching. The 30--31B
checkpoints default to one 80 GB H100, with a conservative 2,048-token context
limit and four active sequences to preserve KV-cache headroom. GPT-OSS-120B
uses its official MXFP4 checkpoint and also defaults to one 80 GB H100, with a
2,048-token context limit and eight active sequences. Install vLLM in the
separate SimBench environment; this is optional for all other repository
functionality:

```bash
pip install -U "vllm>=0.19.0" mistral-common
```

`--model gpt-6-luna` is a separate cloud-backed judge. It uses WayFlow's OCI
GenAI adapter with the user's OCI config and `OCI_GENAI_COMPARTMENT`; it does
not download weights, use a local GPU, or use vLLM. It sends the same blinded
prompt variants and writes exactly the same raw-variant and consensus JSONL
format as the local judges. GPT-6 Luna's OCI endpoint uses its provider-default
temperature, so the run records `temperature=1.0`; `--seed` is still passed to
OCI and retained in every cache key. Use a 512-token cap so a valid JSON answer
is not squeezed out by model reasoning.

Run the same frozen catalog through each judge. Pin a Hugging Face commit hash
rather than `main` before treating outputs as frozen qrels. The script does not
assume a user-specific cache location: it respects standard Hugging Face cache
settings (`HF_HOME`/`HF_HUB_CACHE`) and Hugging Face's normal defaults. To put
model and runtime caches on storage, set `SIMBENCH_MODEL_CACHE_ROOT` to a path
that is accessible to the Ray workers; the script places Hugging Face, vLLM,
PyTorch, and XDG caches beneath it. Explicit settings such as `HF_HOME` still
take precedence. Because Ray workers may run under a service account, use an
absolute path that the worker can read, rather than relying on `$HOME` or the
worker's `$USER` to identify the submitting user's cache.

```bash
export SIMBENCH_MODEL_CACHE_ROOT="/storage/${USER}/model-cache"

# The relationship-only clarification and paired plain-language candidate
# explanation and full prompt paraphrases are prompt version v7. Keep these
# labels separate from v4-v6.

python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model mistral \
  --model-revision main \
  --batch-size 16 \
  --output annotations/rel-amazon/llm/signal-relevance-v7/mistral-small-3.2.jsonl

python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model gpt-oss-120b \
  --model-revision main \
  --batch-size 8 \
  --max-tokens 512 \
  --output annotations/rel-amazon/llm/signal-relevance-v7/gpt-oss-120b.jsonl

python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model gpt-6-luna \
  --batch-size 4 \
  --max-tokens 512 \
  --output annotations/rel-amazon/llm/signal-relevance-v7/gpt-6-luna.jsonl \
  --metrics-output results/rel-amazon/llm-judge/signal-relevance-v7/gpt-6-luna-metrics.json
```

Before a full run, benchmark steady-state annotation speed without writing any
judgments. The printed/filed metrics separate model-load and warm-up time from
prompt throughput, and include prompt/output token rates and JSON validity.

```bash
python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model mistral \
  --batch-size 16 --benchmark-only --benchmark-pairs 64 --warmup-pairs 8 \
  --metrics-output results/rel-amazon/llm-judge/profiles/mistral-small-3.2-throughput.json

python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model muse-glimmer-30b \
  --batch-size 4 --benchmark-only --benchmark-pairs 64 --warmup-pairs 8 \
  --metrics-output results/rel-amazon/llm-judge/profiles/muse-glimmer-30b-throughput.json

python scripts/judge_signals.py \
  --conditions conditions/rel-amazon/conditions-v3.jsonl \
  --dataset-root data/dev/rel-amazon \
  --model gemma-4-31b \
  --batch-size 4 --benchmark-only --benchmark-pairs 64 --warmup-pairs 8 \
  --metrics-output results/rel-amazon/llm-judge/profiles/gemma-4-31b-throughput.json

```

The judge uses a strict JSON Schema: exactly the top-level fields `status`,
`grade`, and `reason` are allowed. The v7 prompt uses three full, equivalent
instruction variants rather than changing only the opening sentence. It
presents each candidate with
both its canonical technical definition and the same plain-language explanation
shown by the human annotation tool. The v7 prompt also explicitly defines a signal
as evidence for a downstream retriever or ranker, rather than as a complete
retrieval procedure. It infers the retrieval target from the intent itself and
grades the complete source-to-signal path, while treating a direct zero-hop
attribute of records named in the intent as valid evidence. The rubric is: `0 = no plausible useful
evidence`, `1 = weak or indirect`, `2 = meaningfully useful but incomplete,
noisy, general, or not central`, and `3 = central and direct`.

`--reasoning-effort` is versioned in the cache key. GPT-OSS supports `none`,
`low`, `medium`, and `high`; GPT-6 Luna accepts `none`, `low`, `medium`,
`high`, and `xhigh` through OCI. Use `none` for concise structured judgments.
Mistral, Muse Glimmer, and Gemma use `none`.

After human calibration, open
[`notebooks/analyze_signal_discovery_results.ipynb`](notebooks/analyze_signal_discovery_results.ipynb)
to browse discovery rankings and compare them interactively against human or
LLM qrels. It reports qrel coverage and does not count unjudged candidates as
irrelevant.

LLM judgments are scalable weak qrels, not human ground truth. The detailed
rubric, audit protocol, and model-selection rationale are in
[`docs/LLM_AS_A_JUDGE_DESIGN.md`](docs/LLM_AS_A_JUDGE_DESIGN.md).

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
