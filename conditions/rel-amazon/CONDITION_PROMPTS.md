# RelAmazon condition prompts

Generate five conditions total for the whole dataset, not five per entity type.
The generator receives the schema. It should use natural user vocabulary rather
than mechanically repeating field names. The separate fit check must not
rewrite proposals.

## Generate five conditions

```text
Create exactly 5 distinct natural-language intent conditions for a benchmark over this relational dataset.

Schema:
- product: product_id, category, brand, title, description, price
- customer: customer_id, customer_name
- review: review_time, customer_id, product_id, rating, verified, review_text, summary
- review.customer_id references customer.customer_id
- review.product_id references product.product_id

The five conditions are for the dataset as a whole. Each request may involve one table or connect multiple tables.

These conditions will be evaluated with a retrieval and ranking system that ranks individual records by relevance to a request. Write requests where records can be judged more or less relevant; do not ask for aggregate statistics, trend reports, or explanations.

Make the five requests meaningfully varied and broad. Write them as natural user requests, using varied vocabulary rather than mechanically repeating schema field names. Requests may involve one table or multiple related tables; do not force a fixed mix of result types or query patterns.

Do not anchor a request to a particular value or narrow, highly specific attribute. 

Return only JSONL, exactly 5 objects, one per line. Each object must have these fields: id, condition. Use IDs condition-001 through condition-005. Do not add explanations or Markdown fences.
```

## Check dataset fit

Run this prompt after generation. You may provide a summary of data coverage
and relationship counts along with the schema if available.

```text
Check these five intent conditions for whether they can be evaluated on the dataset described below. This is a first-pass feasibility check, not manual approval. Do not rewrite the conditions.

Schema:
- product: product_id, category, brand, title, description, price
- customer: customer_id, customer_name
- review: review_time, customer_id, product_id, rating, verified, review_text, summary
- review.customer_id references customer.customer_id
- review.product_id references product.product_id

Use any data coverage and relationship summary supplied with this prompt.

For each condition, assign one decision:
- KEEP: its intent is supported by the schema and available data;
- UNCERTAIN: schema support is plausible, but data coverage or relationship support needs checking;
- REMOVE: it depends on information or a relationship the dataset does not provide.

Base the assessment on fields and relationships actually present. Do not assume that an unstructured text field reliably supports a specific attribute, or that recorded reviews represent every customer interaction.

Do not rewrite, improve, shorten, rank, or drop any condition. Copy id, result_entity_type, intent_family, and intent exactly. Add only decision and a brief reason. Return exactly 5 JSONL objects and no other text.

The five generated JSONL condition objects follow:
```
