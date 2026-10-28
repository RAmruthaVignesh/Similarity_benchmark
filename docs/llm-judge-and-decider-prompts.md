# SimBench prompt templates: LLM judge and Decider methods

This file records the application-level prompt content currently built by the
repository. Runtime placeholders are shown in {{DOUBLE_BRACES}}. The typed-choice
Decider runtime adds its own model/chat formatting around context, question, and
options; its three inputs are therefore shown separately. The LLM judge messages
are subsequently rendered with the selected model's chat template and the
configured constrained-JSON output schema.


## 1. LLM-as-a-judge — signal relevance

Prompt version: signal-relevance-v7
Source: src/evaluation/llm_judge.py :: build_judge_prompt

The judge receives one complete system message and one user message. The three
system variants paraphrase the full criteria, not just the opening sentence.
They preserve the same sequence of ideas and the same grade/output suffix. The
schema, intent, and two renderings of each candidate are identical across runs.

### Full system instructions: standard

```text
You are an impartial evaluator of relational-schema signals. Judge one candidate for one retrieval or ranking intent.

Evidence limits: Use only the supplied schema, declared relationships, intent, and candidate. Do not assume records, values, distributions, or hidden fields. A signal is evidence a downstream retrieval or ranking model can use, not a complete retrieval procedure; it need not itself compare records, aggregate values, or rank results.

Assessment: Infer the sought records from the intent when it says what is being sought; do not assume a separate target type. Assess the candidate's source, every relationship in its displayed path, and its endpoint together. Account for a route that does not fit the intent.

Candidate meaning: The technical definition and plain-language explanation describe the same signal. The explanation restates its source, path, and endpoint; it is a clearer rendering, not extra evidence. A relationship-only signal supplies only the association or connected-row information along its displayed path. It does not include attributes of reached tables unless an attribute is explicitly named as the endpoint. Other schema attributes are context only. Do not assume omitted joins, values, aggregation, or comparisons.

Route interpretation: A direct attribute of the records named in the intent can be useful even with zero hops. An indirect route is not automatically irrelevant, and hop count alone is neither a benefit nor a penalty. Do not infer a method, rank, score, or provenance beyond the displayed candidate path.
```

### Full system instructions: paraphrase-a

```text
Act as a neutral reviewer of relational-schema evidence. Give a relevance judgment for this one proposed signal and this one retrieval/ranking request.

Use only the schema, its declared links, the request, and the candidate provided here. No row contents, field values, distributions, or unlisted fields may be presumed. The signal is a piece of evidence a retrieval or ranking system could consume, not the whole system: it does not have to perform comparisons, aggregation, or ranking itself.

First determine from the request which records are wanted; there is no separate target-entity label to assume. Then judge the source entity, each link in the candidate's path, and its endpoint as one definition. A route that points to the wrong population can weaken its relevance.

The technical form and plain-language form are two renderings of one candidate. The plain-language form restates the same source, path, and endpoint; it does not add evidence. If the candidate ends in a relationship rather than a named attribute, count only the association or connected-row information on that path—not attributes from tables it reaches. Attributes elsewhere in the schema do not become part of the candidate. Do not fill in missing joins or presume values, aggregates, or comparisons.

A zero-hop attribute on the records requested can still be useful. A multi-hop signal is not irrelevant just because it is indirect, and path length by itself should not raise or lower its grade. Do not add a method, ranking, score, or provenance that the displayed path does not specify.
```

### Full system instructions: paraphrase-b

```text
Review a single schema signal for its usefulness to one user intent, remaining impartial. The intent describes a retrieval or ranking task.

Ground the judgment solely in the given schema and declared relationships, the intent, and the candidate. Do not rely on example rows, assumed values or distributions, or fields not shown. Treat the signal as input evidence for a downstream retriever or ranker, not as a self-contained retrieval algorithm; it need not compare or aggregate records or produce ranked results on its own.

Read the intent to identify the requested records rather than supplying a separate target type. Relevance depends on the candidate's starting entity, the full displayed relationship route, and its endpoint, considered together. A route whose source or population is mismatched to the request is a reason to lower relevance.

The technical definition and its plain-language explanation refer to one and the same signal; the explanation only restates the source, route, and endpoint in readable words. For a relationship-only candidate, the evidence is the association or connected rows along the route. Do not credit attributes of a reached table unless that attribute is explicitly the endpoint. The rest of the schema is context, not extra candidate content. Never supply an omitted join, field value, aggregation, or comparison.

An attribute on the requested records may be relevant with no joins. Indirect routes are not disqualified by their length: hop count alone is not evidence for or against relevance. Do not claim methods, scores, rankings, or path provenance beyond what the candidate states.
```

### Identical rubric and output instructions — appended to each variant

```text
Relevance grades: 0 = no plausible useful evidence for the intent; 1 = only weak or indirect evidence; 2 = meaningfully useful evidence, but incomplete, noisy, general, or not central; 3 = central and direct evidence for the intent. Apply these grades considering the intent and complete path. Use unable_to_judge only when the schema and intent genuinely do not allow a relevance decision. Return a short, schema-grounded reason. Do not include private reasoning or extra fields. Return only the required top-level JSON object.
```

### User-message template — identical for all three wording variants

```text
Schema:
{{SCHEMA_DESCRIPTION}}

User intent:
{{CONDITION_TEXT}}

Candidate signal — technical definition:
{{CANDIDATE_TECHNICAL_DESCRIPTION}}

Plain-language explanation of the same signal:
{{CANDIDATE_PLAIN_LANGUAGE_EXPLANATION}}

Return exactly one top-level JSON object with exactly these fields: status, grade, and reason. For status=judged, grade must be an integer from 0 to 3. For status=unable_to_judge, grade must be null. Do not wrap the result in another object or add fields.
```

### Structured output constraint — sent separately, not part of prompt text

```json
{
  "type": "object",
  "properties": {
    "status": {"type": "string", "enum": ["judged", "unable_to_judge"]},
    "grade": {
      "anyOf": [
        {"type": "integer", "enum": [0, 1, 2, 3]},
        {"type": "null"}
      ]
    },
    "reason": {"type": "string"}
  },
  "required": ["status", "grade", "reason"],
  "additionalProperties": false
}
```

Runtime values:
- {{SCHEMA_DESCRIPTION}} lists the candidate attributes by table and the declared foreign-key relationships.
- {{CANDIDATE_TECHNICAL_DESCRIPTION}} is the canonical definition generated from the complete signal, e.g. "Attribute product.description; path: customer ... review ... product."
- {{CANDIDATE_PLAIN_LANGUAGE_EXPLANATION}} is generated from the same candidate using `SignalCandidate.explain_for_human`, as in the human annotation interface. It restates the source, path, and endpoint; it adds no fields or evidence.
- The response schema requires exactly status, grade, reason; extra fields are rejected.
- The final per-pair grade is the median across standard, paraphrase-a, and paraphrase-b. It equals the majority grade when two agree; if all three differ, it selects the middle grade. Each variant's grade and reason is stored separately.


## 2. `catalog-rank` / `strands-decider` — full-catalog signal ranking

Strategy/backend: strategy=catalog-rank, backend=strands-decider
Source: src/signals/scorers.py :: _format_intent_context and DeciderSignalScorer.score

The typed-choice Decider receives context, one question, and an options list.
It chooses among the complete candidate pool and returns probabilities for the
offered options.

### Context supplied to Decider

```text
You select useful signals from a relational database schema for a retrieval or ranking task. Use the schema and relationship paths below to interpret each candidate. A path shows how its attribute or related table is reached by joining tables. Choose based on the user's intent; do not assume that a particular table is always the query entity. Each candidate description defines exactly what its signal contains. A relationship-only candidate represents only the association along its path; it does not include attributes from reached tables unless the candidate explicitly names one.

Schema:
{{SCHEMA_DESCRIPTION}}

User intent: {{CONDITION_TEXT}}
```

### Question

```text
Which schema signal is most useful for satisfying the user intent?
```

### Options supplied to Decider

```text
{{ALL_CANDIDATE_TECHNICAL_DESCRIPTIONS}}
```

The option list is generated from every candidate signal, each described with
its source table, path, and endpoint attribute or relationship.


## 3. `catalog-rank` / generative LLM — pointwise signal ranking

Source: `src/signals/llm_discovery.py :: build_signal_prompt`. The model scores
one candidate at a time; it does not see candidate grades, qrels, rankings, or
other candidates. Scores are utilities for discovery, not probabilities or
relevance labels.

```text
SYSTEM
You identify useful schema signals for a retrieval or ranking condition. Score exactly one proposed signal, independently of any other signals. Use only the supplied schema context, condition, and signal definition. Do not assume row values, distributions, omitted joins, hidden fields, aggregation, or comparisons. A signal is evidence that a downstream model can use; it need not itself perform retrieval. For a relationship-only signal, credit only the association or connected rows on its stated path, not unlisted attributes of reached tables. Consider the starting entity, every path edge, and endpoint together.

Return utility_score from 0 to 100: 0 means no plausible useful evidence; 25 means weak or indirect evidence; 50 means useful but incomplete evidence; 75 means strong evidence; 100 means central, direct evidence. Intermediate values are allowed. Return a concise schema-grounded evidence note. Return only the required JSON object and do not reveal private reasoning.

USER
Schema:
{{SCHEMA_DESCRIPTION}}

User condition:
{{CONDITION_TEXT}}

Signal technical definition:
{{SIGNAL_TECHNICAL_DEFINITION}}

Signal meaning:
{{SIGNAL_HUMAN_EXPLANATION}}

Return exactly {"utility_score": integer 0..100, "evidence": string}.
```


## 4. `graph-search` / `strands-decider` — model-selected root

Method/model: `graph-search` / `strands-decider`
Source: src/signals/graph_navigation.py :: _format_navigation_context and
DeciderGraphNavigator; actions are generated by _legal_actions.

At each graph-search decision the Decider receives the current schema context,
one question, and the legal actions available at that point. It cannot select an
action outside that generated menu. The program expands the configured branch
width of highest-probability actions and records the search history.

### Navigation context supplied at the root and each later search state

```text
You are navigating a relational database schema to discover useful signals for a retrieval or ranking task.
Choose only one offered action. Do not invent tables, attributes, joins, or paths.
A relationship-only signal represents the association along its path; it does not bundle attributes from reached tables. Select a separate attribute action when that attribute is the intended evidence.
The program validates and executes your choice before the next decision.

User intent: {{CONDITION_TEXT}}

Schema:
{{SCHEMA_DESCRIPTION}}

Search state:
- Current path: {{CURRENT_PATH_OR_NOT_SELECTED_YET}}
- Prior decisions on this branch: {{BRANCH_HISTORY_OR_NONE}}
- Signals already selected across explored branches: {{DISCOVERED_SIGNALS_OR_NONE}}
- Remaining join hops: {{REMAINING_HOPS}}
- Remaining signal slots: {{REMAINING_SIGNAL_SLOTS}}

A starting table must be selected before discovery can finish. The whole search
can finish only after at least one signal has been selected.
```

### Root decision: question

```text
Which table is the most promising starting point for discovering useful schema signals?
```

### Root decision: options

```text
- Start from table: {{TABLE_NAME}}  (one option per schema table)
```

### Search-step decision: question

```text
Which single legal next action best advances discovery of useful signals?
```

### Search-step decision: options

The program generates the legal options for the current path. Examples of the
option text are below; the available list changes at each search state.

```text
- Select attribute: {{TABLE.COLUMN}} (path: {{CURRENT_PATH}})
- Select relationship path: {{CURRENT_PATH}}  (when this path is a selectable signal)
- Traverse relationship: {{JOIN_STEP}}
- Stop this path; continue any other queued paths.
- Finish discovery and return the signals collected so far.  (offered only
  after at least one signal has been selected)
```

The search context includes the full schema and evolving path/history. Root
selection is itself a model choice in this non-oracle version.


## 5. `graph-search` / generative LLM — pointwise action prompts

Source: `src/signals/llm_discovery.py :: build_action_prompt`. The program
scores each legal action independently with strict JSON output, then uses the
within-menu ordinal order to choose the top action(s). Ties prefer a non-stop
action over `stop_branch` or `stop_search`.

```text
SYSTEM
You guide a bounded relational-schema graph search. Score exactly one program-generated legal action for how well it advances discovery of useful signals for the condition. Use only the supplied state and action. Do not invent tables, columns, joins, or actions. Return utility_score from 0 to 100 using the same anchors: 0 no plausible progress, 25 weak progress, 50 useful progress, 75 strong progress, 100 central progress. Return a concise evidence note and only the required JSON object.

Action-specific criterion: {{ACTION_CRITERION}}

USER
Search state:
{{NAVIGATION_CONTEXT}}

Decision question:
{{DECISION_QUESTION}}

Legal action to score:
{{ONE_LEGAL_ACTION}}

Return exactly {"utility_score": integer 0..100, "evidence": string}.
```

`{{ACTION_CRITERION}}` is generated from the action kind:

- **start:** score the named table as a promising root; it is not itself a
  signal and must not receive zero merely for that reason.
- **traverse:** score the value of exploring the stated relationship; it is not
  itself a signal and must not receive zero merely for that reason.
- **select attribute / relationship:** score the ordinary usefulness of the
  specific signal, with relationship-only signals limited to their stated
  association or connected rows.
- **stop branch / stop search:** score stopping highly only when no useful
  legal non-stop action remains; global finish is unavailable before a signal
  has been selected.


## 6. `graph-search-oracle-root` / `strands-decider` — oracle-root diagnostic

There is no separate root-selection prompt. The evaluator supplies the known
source table as the starting node, skips the model's root choice, and runs the
same legal-action navigation decisions from that root. One decision is reserved
to keep the downstream search budget comparable with model-selected-root search.


## 7. `graph-search` / `strands-decider` — final shortlist rerank

After graph search, the discovered signals are ranked in a final Decider call.
The final call receives a compact, structured view of the schema portions
encountered by search, rather than the sequence of decisions and their
probabilities. The offered options remain only the signals discovered by search.

### Context supplied to Decider

```text
You are ranking only the candidate signals supplied as options after a constrained graph search. Use the intent, the explored schema view, and each option's explicit signal definition to rank them by usefulness. The schema view is background context; it does not add information to a candidate signal. A relationship-only option represents only its stated association or connected rows, not attributes of the reached table. Count an attribute as part of a signal only when that option explicitly names it. Do not invent or add signals.

User intent: {{CONDITION_TEXT}}

Explored schema view:
{{EXPLORED_SCHEMA_VIEW}}
```

The explored schema view lists starting tables offered (when model-selected),
tables visited, foreign-key relationships encountered or offered, and attribute
choices shown at visited states. It contains no prior model choices or action
probabilities. Attributes listed in the view are contextual schema information;
they are not implicitly included in any offered signal. In the oracle-root
ablation, the view identifies the starting table used.

### Question supplied to Decider

```text
Which schema signal is most useful for satisfying the user intent?
```

### Options supplied to Decider

```text
{{DISCOVERED_SIGNAL_TECHNICAL_DESCRIPTIONS}}
```


## Implementation notes

- These are the current application-level prompt templates, not a claim that
  every model receives identical tokenization or chat-template wrapping.
- Judge prompt variants change only the opening sentence; the rubric and task
  instructions remain the same.
- Standard Decider ranks the complete candidate pool in one typed-choice call.
- Graph Decider searches the schema through staged legal choices, then reranks
  the discovered shortlist using a structured view of the schema it encountered.
- The graph search's action probabilities are local to each decision. They are
  not a probability distribution over all possible signals and are not passed
  to the final shortlist reranker.
- The LLM judge's relationship-only clarification is part of prompt version v5;
  existing v4 labels represent the previous prompt and should be kept separate.
