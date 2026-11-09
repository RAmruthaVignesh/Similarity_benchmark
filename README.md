# SimBench

SimBench is a benchmark and prototype pipeline for conditioned similarity over
relational data. A user condition is used to discover useful schema signals,
then the selected signals are materialized for entities and combined into a
similarity ranking.

The current development dataset is a local RelAmazon subset with `product`,
`customer`, and `review` tables. The benchmark contains 15 conditions and
complete human signal-relevance labels.

## What the repository contains

```text
conditions/rel-amazon/             Conditions and the oracle-root diagnostic map
annotations/rel-amazon/human/      Human labels for signal discovery
data/dev/rel-amazon/               Local RelAmazon subset (generated; ignored)
results/rel-amazon/discovery/      Saved signal-discovery rankings
results/rel-amazon/similarity/     Entity-similarity runs and analyses
src/signals/                       Signal catalogues and discovery methods
src/similarity/                    Signal materialization and similarity measures
scripts/                           Command-line entry points
notebooks/                         Analysis and exploration notebooks
docs/                              Prompts and experimental-scoring details
```

`data/` and most result directories are local derived data. The condition and
human-label files are benchmark inputs.

## Core concepts

A **condition** describes the requested kind of similarity, for example
“find products with similar purpose and features.” A **signal** is one piece of
schema information that may help answer it.

There are two signal types:

- An **attribute signal** ends at a column, such as
  `product → review → review.rating`.
- A **relationship signal** uses the set of reachable related entities itself,
  such as `product → review → customer`.

The path is part of the signal. It specifies both the entity from which the
signal is collected and the relationships followed. For example,
`product → product.description` gives one description per product, whereas
`customer → review → product → product.description` gives each customer a
collection of descriptions for products they reviewed.

The standard catalog enumerates direct attributes, relationship paths, and
attribute paths through up to two foreign-key hops. The path-aware catalog
keeps these candidates distinct. A feature-collapsed view is also available
for evaluation, where provenance variants of the same physical attribute are
collapsed.

## Main workflow

1. **Discover signals.** `catalog-rank` presents a model with one condition
   and one candidate signal at a time. The model receives the schema context,
   the human-readable path meaning, and the technical path definition. It
   returns an integer utility score on the default 0–10 scale and brief
   schema-grounded evidence. Candidates are ranked by utility score, with a
   deterministic candidate-ID tie-break. The score is a ranking utility, not
   a probability or a human relevance label.
2. **Select compatible signals.** To rank entities in a table, select signals
   rooted at that same table. Selection can use the highest-ranked signals or
   every compatible signal above a 0–10 utility threshold.
3. **Materialize and score.** For each source entity, follow each selected
   path, collect its values, compare those values to every candidate entity,
   and combine the component similarities.
4. **Evaluate.** Signal-discovery runs are evaluated against the independent
   human labels with nDCG. Entity-similarity rankings can be inspected in the
   notebooks and compared through blinded human pairwise comparisons.

The standard discovery method is catalog ranking. `graph-search` and
`graph-search-oracle-root` are retained as graph-search comparisons; the
oracle-root version is diagnostic only.

## Similarity measures

`SignalMeasureExecutor` chooses a measure from the signal values:

| Signal value | Default measure |
| --- | --- |
| Text or text collections | E5 embedding cosine similarity |
| Scalar numeric or temporal value | Percentile-distance similarity |
| Numeric or temporal collection | Percentile Wasserstein similarity |
| Scalar category or boolean | Exact agreement |
| Categorical collection | IDF-weighted Jaccard similarity |
| Boolean collection | Boolean-rate similarity |
| Relationship ending at stable entity IDs | Neighbourhood Jaccard similarity |
| Relationship without stable endpoint IDs | Degree similarity |

Path-based collections receive lower effective weight when either entity has
little supporting evidence. Direct scalar attributes retain full weight. By
default, unavailable component scores are omitted from the weighted mean.

An optional bounded robust-z profile is available for analysis. It normalizes
each signal using held-out-query training scores, keeps scores in `[0, 1]`,
and handles missing evidence with an explicit reference value and reported
evidence coverage. See
[experimental-similarity-scoring.md](docs/experimental-similarity-scoring.md).

## Running the pipeline

Run commands from the repository root in an environment with the project
dependencies installed. Activate it using the name or path configured for your
machine, for example:

```bash
conda activate simbench
```

The Hugging Face libraries use their standard cache by default. If your team
uses a shared cache, configure `HF_HOME` or `HF_HUB_CACHE` in your own shell
environment rather than relying on a machine-specific path in this repository.

Inspect current options rather than relying on old commands:

```bash
python scripts/discover_signals.py --help
python scripts/score_entity_similarity.py --help
```

Example catalog-ranking run using the default local paths:

```bash
python scripts/discover_signals.py \
  --method catalog-rank \
  --model gpt-6-luna \
  --utility-scale 0-10 \
  --catalog-view path-aware \
  --output results/rel-amazon/discovery/catalog-rank-gpt-6-luna-path-aware.jsonl
```

Example entity-similarity run using selected signals from a discovery file:

```bash
python scripts/score_entity_similarity.py \
  --condition-id condition-001 \
  --entity-table product \
  --discovery-results results/rel-amazon/discovery/catalog-rank-gpt-6-luna-path-aware.jsonl \
  --min-utility-score 5 \
  --query-id '<PRODUCT_ID>' \
  --output results/rel-amazon/similarity/example.jsonl
```

For a text signal, `score_entity_similarity.py` loads the local E5 model and
uses its persistent embedding cache. `scripts/precompute_text_embeddings.py`
can populate that cache before interactive use.

## Notebooks and tools

- `notebooks/analyze_signal_discovery_results.ipynb`: evaluate and browse
  discovery rankings against human labels.
- `notebooks/compare_catalog_rank_runs.ipynb`: compare catalog-rank model and
  reasoning-effort runs.
- `notebooks/analyze_graph_search_results.ipynb`: inspect graph-search runs.
- `notebooks/explore_entity_similarity.ipynb`: interactively inspect selected
  signals, component scores, and nearest entities.
- `notebooks/demo_end_to_end_similarity.ipynb`: enter a condition, discover
  signals, and score an entity in one notebook.
- `notebooks/compare_conditioned_entity_rankings.ipynb`: compare original and
  bounded robust-z SimBench rankings, plus any available external baseline.
- `notebooks/analyze_rel_amazon_subset_eda.ipynb`: inspect the local subset,
  entities, and relational structure.
- `scripts/explore_similarity_ui.py`: lightweight browser alternative to the
  notebook explorer.

## Documentation and checks

The exact discovery prompts are in
[docs/signal-discovery-prompts.md](docs/signal-discovery-prompts.md). The
experimental scoring protocol is in
[docs/experimental-similarity-scoring.md](docs/experimental-similarity-scoring.md).

Run the offline test suite with:

```bash
python -m unittest discover -s tests
```
