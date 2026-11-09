# Current workflow

## 1. Build the signal catalog

For the relational schema, we generate candidate signals from every
table and simple foreign-key path up to two hops.

- An attribute candidate ends at a column, for example
  `product → review → review.rating`.
- A relationship candidate ends at reachable entities, for example
  `product → review → customer`.

The path is kept with the signal because it changes what one entity contributes
to the comparison. A direct product description is one value for a product;
a customer-to-review-to-product-description path is a collection of product
descriptions associated with that customer.

## 2. Rank signals for a condition

Given a condition (for example, "Find product with similar purpose and price"), we then want to select the relevant signals.
The main discovery method is **catalog ranking**.

For every condition–signal pair, an LLM sees:

- the relational schema context;
- the condition text;
- one candidate signal’s.

It returns a 0–10 utility score and a short explanation of the evidence in the
schema. The model does not see human labels, other candidates, relevance
grades, or the contents of the tables.

The 0–10 score is used either to take the top-ranked signals.

Graph-search methods are also implemented for comparison. They let a model explore the schema step by step, then rerank the signals it discovers. Catalog ranking is currently the primary method because it performed best against the human annotations.

## 3. Select signals for an entity-ranking task

To compare entities in a table, we use only signals rooted at that table. For
example, to find products similar to one product, selected signals must start
at `product`. This ensures each candidate can be materialized once for every
product and compared consistently to the reference product.

## 4. Turn each selected signal into pairwise similarity

For a reference entity and a candidate entity, the executor follows the
signal’s path from each entity, collects the endpoint value or values, and
applies a type-appropriate similarity measure.

| Data reached by a signal | Default comparison |
| --- | --- |
| Text, including collections such as review text | embedding cosine similarity |
| One numeric or date/time value | Difference after mapping values to their dataset percentiles |
| A collection of numeric or date/time values | Wasserstein similarity over percentile-normalized values |
| One categorical or boolean value | Exact agreement |
| A collection of categories | IDF-weighted Jaccard overlap |
| A collection of booleans | Similarity of true-value rates |
| Related entities with stable IDs | Jaccard overlap of neighbour sets |
| Related rows without stable endpoint IDs | Similarity of relationship degree |

Text embeddings are currently generated with a local E5 pooling model.

## 5. Combine signals into the final rank

Each selected signal produces a component score between 0 and 1. The original
SimBench score is the weighted mean of available component scores. Signal
weights are equal by default and can be changed in the explorer or command
line.

We also have an experimental bounded robust-z aggregation mode to investigate
whether heterogeneous measures are on comparable scales. It fits a median and
robust scale for each signal on training reference queries, transforms scores
monotonically to `[0, 1]`, and keeps missing evidence in the denominator at a
signal-specific reference score.

The output for each query contains the ranked candidates, final similarity,
and the per-signal raw score, effective weight, confidence, and measure.

## 6. Current evaluation

[Signal selection](#2-rank-signals-for-a-condition) is evaluated with nDCG against complete human labels for 15 conditions on RelAmazon.

Key entry points:
- `notebooks/analyze_signal_discovery_results.ipynb` for signal selection evaluation.
- `notebooks/explore_entity_similarity.ipynb` and
  `notebooks/demo_end_to_end_similarity.ipynb` for interactive exploration.
- `notebooks/compare_conditioned_entity_rankings.ipynb` for original versus
  bounded robust-z comparison.
