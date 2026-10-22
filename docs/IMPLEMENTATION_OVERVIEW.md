# SimBench: current implementation overview

RelAmazon is the current development dataset. This document summarizes the
implemented signal-discovery (identify and rank useful schema signals) and evaluation work, an initial component of the
broader goal: evaluating relational models that embed entities according
to a user's natural-language condition.


## Signal catalog

There are two types of candidate signals:

- **Feature-centered signals:** It consists in a feature and a path: use a feature (an attribute/column) from the
  starting table or a table reached by following relationships. These can be
  zero-hop (the feature is on the starting table) or multi-hop.
- **Relationship signals:** use the relationship path itself as evidence,
  without selecting an endpoint feature.

With the standard RelAmazon settings, the catalog contains 39 candidates from
all tables and paths up to two foreign-key hops. A path specifies how the signal is reached. For example
product.description reached directly from the product table (path: product → product.description) is considered as a distinct signal to product.description reached from path customer → review → product → product.description 
In this example the path changes the signal’s meaning:
- Starting from product gives one description for that product.
- Starting from customer and following reviews to products gives that customer a collection of descriptions (the descriptions of products they reviewed). That can summarize the customer’s interests.

## Signal-discovery methods

| Method | Summary |
| --- | --- |
| Decider full-catalog | A typed-choice Decider receives the condition, schema context, and complete candidate list, then ranks/selects signals from that catalog. This is the one-shot schema-reasoning method. |
| Decider graph | Searches the schema through choices, then reranks the discovered shortlist using the condition and an explored-schema view. The final reranking is not over the full catalog. |
| Decider graph, oracle-root | Runs graph search from the source table specified for that condition, rather than asking Decider to choose the starting table. This controlled-only diagnostic isolates the effect of root selection. |

In more details, the ordinary **Decider graph** method works as follows:

1. **Choose a starting table.** Decider chooses among the schema tables. It can retain multiple high-probability starting
   tables, rather than forcing the search to keep only one.
2. **Explore the graph and collect signals.** At each state, Decider chooses
   among legal actions: follow a foreign-key relationship, select an available
   feature or relationship signal, stop a path, or finish. The program checks
   each action against the schema and applies the hop and decisions.
3. **Rerank the discovered signals.** A final Decider call ranks only the
   selected shortlist, using the condition and the explored-schema view. It does
   not receive the full schema or undiscovered signal candidates at this stage.



The **oracle-root** variant skips the model's starting-table choice and uses
the source table specified for the controlled condition, then follows the same graph-search
and final-reranking stages. 




All Decider methods use the model documented in the [Strands Decider E4B model card](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610).

## Controlled synthetic signal benchmark

The controlled suite contains 25 simple authored conditions over the real RelAmazon
schema and standard 39-candidate catalog. These conditions vary the reasoning
requirements, while their graded references are explicitly authored in code.


An early result snapshot for Strands Decider E4B is below. The metric shown requires the exact source table and relationship
path in the authored reference to match.

| Method | nDCG@5 |
| --- | ---: |
| Decider full-catalog | 0.685 |
| Decider graph | 0.514 |
| Decider graph, oracle-root | 0.968 |

These are preliminary, author-defined synthetic results. The performance of the Decider graph method can be explained by the fact that it selects the right features most of the time but with the wrong corresponding paths.
## LLM-as-a-judge

The judge evaluates one condition–candidate pair at a time using the condition,
full path-specific signal, and schema context. It is blinded to discovery
method, model rank/score, model-generated explanation, and graph-search history.
It does not read source row values. The ordinal rubric is 0 (no plausible
useful evidence), 1 (weak/indirect), 2 (meaningfully useful but incomplete or
not central), and 3 (central/direct); it can also return `unable_to_judge`.

Implemented judge backends:

- local vLLM: Mistral Small 3.2, Muse-Glimmer 30B, Gemma 4 31B, and GPT-OSS
  120B;
- OCI WayFlow: `gpt-6-luna`.

For every pair, three near-paraphrase prompt variants use the same criteria and
user message. The final grade is the median of the three grades;
each variant's grade and reason are retained for audit and prompt-sensitivity
analysis. 

### Research references for the LLM-as-a-judge implementation and next steps

- [NIST, *A Large-Scale Study of Relevance Assessments with LLMs*](https://www.nist.gov/publications/large-scale-study-relevance-assessments-large-language-models-initial-look): LLM judgments reproduced system-level rankings well in a TREC setting, while humans were stricter.
- [Fröbe et al., *LLM Relevance Assessors Agree With One Another More Than With Human Assessors*](https://eprints.gla.ac.uk/352747/): model agreement is not a substitute for independent human validation. LLM assessors can favor LLM-based rankers.
- [Takehi et al., *LLM-Assisted Relevance Assessments (LARA)*](https://arxiv.org/abs/2411.06877): use a limited human-label budget strategically to calibrate and extend LLM judgments.
- [Shi et al., *Judging the Judges: Position Bias in LLM-as-a-Judge*](https://arxiv.org/abs/2406.07791): judgments can vary with prompt variants and repetition, motivating stability checks.
