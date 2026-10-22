# LLM-as-a-Judge for Signal Relevance: Design Proposal

## Purpose

Use an LLM judge to assign provisional relevance grades to condition–signal pairs, then use those grades to compare signal-discovery rankings such as MiniLM, Decider, and Decider-Graph.

This evaluates **signal identification**, not whether a downstream embedding or retrieval model improves. LLM judgments are weak qrels for scalable analysis, not human ground truth. Keep them separate from human annotations and label reported results as **LLM-judged**.

## Evaluation unit and candidate set

One judgment is for one `(condition_id, signal_id)` pair. The judge decides how useful that signal—including its attribute or relationship and its path—is for expressing or satisfying the condition.

For the current five user-authored RelAmazon conditions, judging all 39 schema candidates requires at most 195 pair judgments. The 25-condition controlled synthetic suite would require at most 975 pairs, but it already has author-defined qrels: keep those as the primary reference there, and use LLM judgments only as a comparison/sanity check. In both cases, judge the full catalog, not only the signals returned by a method. This gives every method the same reference set, and a relevant signal missed by graph search remains a miss when its ranking is scored.

For later, larger candidate catalogs, use a judged pool: union the top-ranked signals from every method, then add a random or stratified sample from the remaining catalog. Report judged coverage and do not treat unjudged candidates as irrelevant.

## Judge input and blinding

For each pair, provide:

- the condition text;
- a readable description of the candidate signal, including its path;
- the relevant schema definitions and relationships needed to interpret it.

Do not provide the method name, rank, score, model-generated explanation, or search history. Judge pairs independently rather than asking the judge to rank a list of methods’ outputs. This reduces method- and position-related cues.

The first version should use schema information only, not source records. Judge whether the signal is a plausible and useful way to represent the intent; do not assume that a pattern or value distribution exists in the data. Return `unable_to_judge` if the schema and condition do not provide enough evidence. If later conditions genuinely require evidence about observed data, make that a separately defined evaluation mode rather than silently adding records to the current prompt.

## Relevance rubric

Use an ordinal grade:

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant or does not support the condition |
| 1 | Weak or indirect connection |
| 2 | Useful, but not central |
| 3 | Direct or central signal for the condition |
| — | `unable_to_judge`; not a relevance grade |

Ask for a short, schema-grounded reason for auditability. A reason should cite what the signal represents, not reveal hidden chain-of-thought. Treat the output as a judgment, not a probability: model-stated confidence is not assumed to be calibrated.

The primary grade evaluates the **complete path-specific signal** as evidence available to a downstream retrieval or ranking model; it is not a requirement that the signal itself execute a comparison, aggregation, or ranking. When the intent names the records to retrieve, a direct zero-hop attribute of those records is valid evidence. The judge infers the target from the intent itself; no separate target-entity field is provided. Source and path still matter because they specify the provenance and population of that evidence. If alternate provenance paths become a recurring issue, add a secondary diagnostic grade for the underlying attribute/relationship with path ignored; do not conflate that diagnostic with the main path-exact score.

## Scoring methods

Use the same frozen judgments for every method and report macro-averages over conditions:

- **nDCG@k** as the primary metric, since it uses graded relevance and rewards useful signals ranked near the top;
- **Recall@k** to show what fraction of the relevant signals appeared in the top *k*;
- optionally **Precision@k** for the fraction of top-*k* signals judged relevant (predefine which grades count as relevant).

For a small, fully judged catalog, calculate metrics against all judged candidates. With pooled/incomplete judgments, report coverage and use metrics designed for incomplete relevance assessments where appropriate; never score unjudged items as grade 0.

## Implemented version

The design is implemented by three pieces:

- `src/evaluation/llm_judge.py` defines the prompt, rubric and parser, model IDs, cache identity, and vLLM setup.
- `scripts/judge_signals.py` enumerates the full schema candidate catalog, grades each condition/candidate pair, and writes resumable JSONL.
- `notebooks/analyze_signal_discovery_results.ipynb` interactively scores and browses discovery-result JSONL against human or LLM judgments, while reporting qrel coverage.

The local presets are `mistralai/Mistral-Small-3.2-24B-Instruct-2506` (`--model mistral`), `meta-models/Muse-Glimmer-30B` (`--model muse-glimmer-30b`), `google/gemma-4-31B-it` (`--model gemma-4-31b`), and `openai/gpt-oss-120b` (`--model gpt-oss-120b`). They run through vLLM's offline engine. The runner formats each request with the model tokenizer's chat template, batches prompts, enables prefix caching, and constrains output with a JSON Schema that requires exactly the top-level fields `status`, `grade`, and `reason`, with no extra properties. Sampling defaults are temperature 0, seed 0, and 256 generated tokens. The prompt version is `signal-relevance-v7`, so old cached outputs are not reused. It has three complete paraphrases of the assessment guidance; the grade rubric and output contract remain identical. The 30--31B presets use conservative vLLM settings on one 80 GB H100. GPT-OSS-120B uses its MXFP4 checkpoint on one 80 GB H100 with a 2,048-token context cap and eight active sequences. Mistral Small 3.2 uses vLLM's Mistral tokenizer/config/load modes. Install a CUDA-compatible vLLM build in the separate SimBench environment; ordinary data/schema utilities do not import vLLM.

`openai.gpt-6-luna` (`--model gpt-6-luna`) is the cloud comparison judge. It is called through WayFlow's `OCIGenAIModel`, using the caller's OCI configuration and `OCI_GENAI_COMPARTMENT`; it neither downloads local weights nor consumes a local GPU. It receives the identical method-blind three-variant prompts and uses OCI native structured JSON output. The endpoint accepts only its provider-default temperature, so its rows record `temperature=1.0`, and the runner passes the configured seed to OCI. It should use `--max-tokens 512`; the shorter local-model default can leave insufficient room for the visible JSON after reasoning. The cloud runner uses `--batch-size` as the number of concurrent independent OCI requests and emits the same resumable raw-variant and consensus JSONL schema as local runs.

### Prompt templates

The exact full system variants are maintained in
[`llm-judge-and-decider-prompts.md`](llm-judge-and-decider-prompts.md). They
paraphrase the complete assessment criteria while keeping their order, the
grade rubric, and output contract fixed. The user message is:

```text
Schema:
{schema_description}

User intent:
{condition_text}

Candidate signal — technical definition:
{canonical_candidate_description_with_path}

Plain-language explanation of the same signal:
{candidate.explain_for_human(graph)}

Return exactly one top-level JSON object with exactly these fields: status, grade, and reason. For status=judged, grade must be an integer from 0 to 3. For status=unable_to_judge, grade must be null. Do not wrap the result in another object or add fields.
```

`schema_description` lists every table and candidate attribute, followed by the declared foreign-key relationships. `canonical_candidate_description_with_path` is the exact technical definition; `candidate.explain_for_human(graph)` generates the same plain-language explanation displayed in the human annotation interface. The two descriptions represent one signal: the plain-language form restates its source, path, and endpoint and adds no additional evidence. Source table rows and values are not loaded into the judging prompt. The same full schema context is used for every pair, while the candidate and condition change. The prompt does not include discovery output, method identity, score/rank, graph trace, or a candidate list to compare jointly. The parser accepts a grade only when the response is valid JSON with status `judged`, integer grade 0–3, and a nonempty reason; it also accepts `unable_to_judge` with a null grade and nonempty reason. Invalid responses are retried up to the configured attempt count and then retained with `status: invalid_response` for inspection.

The prompt/rubric identifiers currently written to output are `signal-relevance-v7` and `signal-evidence-rubric-1`. The input conditions contain only `id` and `condition`; no target entity is supplied separately. If prompt wording or grade meanings change, increment these versions before creating a new judgment set.

### Candidate set and model loading

The CLI defaults to all tables as source tables, a two-hop path limit, relationship candidates enabled, and structural key columns excluded. This is intended to match the normal 39-candidate RelAmazon catalog. Use the same condition file, dataset schema, source-table selection, and candidate flags for both models. The dataset loader reads local manifest and Parquet schema metadata to enumerate columns; it does not need record values for relevance judgments.

The default model revision is `main`, which is mutable. The runner prints a warning for this setting. For a frozen result, pass a Hugging Face commit hash using `--model-revision`; save the exact vLLM version and command alongside the output. `--tensor-parallel-size` controls GPUs assigned to one model instance; it is not the number of concurrent independent judge processes. `--batch-size` controls prompts passed to vLLM per generation call. Prefix caching can reuse the common schema/system prompt prefix across pairs. `--benchmark-only --benchmark-pairs N` runs N representative pair prompts without producing qrels, reporting model-load seconds, unmeasured warm-up seconds, pairs/s, input/output/total tokens per second, and JSON-response status counts. Use the same batch size, model-length cap, tensor-parallel size, and reasoning setting you intend to use for the real run.

Reasoning effort is explicit and part of the cache key. GPT-OSS supports `none`, `low`, `medium`, and `high`; the other current judge presets use `none`.

### Actual JSONL record and resume behavior

One record is written per pair. A record contains `condition_id`, `condition`, `candidate_id`, the serialized candidate/path, the 0–3 `grade` or null, `status`, the short `reason`, raw model response, parsing error if any, model family/ID/revision, prompt and rubric versions, sampling settings, candidate configuration, schema hash, cache key, attempt number, and timestamp. The key is SHA-256 over the condition ID/text, candidate ID, schema/candidate-catalog hash, model name/revision, prompt/rubric versions, seed, and temperature.

Existing rows are reused only when their cache key matches and their status is `judged` or `unable_to_judge`. Invalid responses are eligible for retry. `--overwrite` regenerates rows with matching keys. Output is written atomically after each full attempt pass, so completed output from a prior pass is preserved; if a process is interrupted partway through its current pass, results produced in that unfinished pass are not yet on disk and will be regenerated on resume. Keep each model's qrels in its own output file.

Example record:

```json
{
  "condition_id": "condition-001",
  "condition": "Find products that serve similar purposes.",
  "candidate_id": "attribute:product:product:product.category",
  "candidate": {
    "id": "attribute:product:product:product.category",
    "kind": "attribute",
    "description": "Attribute product.category; path: product."
  },
  "status": "judged",
  "grade": 2,
  "reason": "Category is useful for grouping products, although it may not capture similarity of purpose by itself.",
  "judge_model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
  "judge_model_revision": "<pinned-commit>",
  "prompt_version": "signal-relevance-v7",
  "rubric_version": "signal-evidence-rubric-1"
}
```

Human annotations remain in separate files and use the existing annotation tool's `status: complete`; the current LLM evaluator reads rows with `status: judged`. Normalize that status before using human annotations with the LLM evaluator, or use the human-annotation comparison workflow. Do not merge LLM and human judgments into one unlabeled qrels file.

### Implemented ranking metrics

The evaluator reads each result's ordered `signals` list and matches by `condition_id` and candidate `id`. It reports a macro-average across conditions of nDCG@1/3/5/10, Recall@1/3/5/10, and Precision@1/3/5/10. nDCG uses all available ordinal grades. Recall and Precision use grade ≥2 as relevant by default; `--relevant-grade` can change that threshold to 1 or 3. The evaluator writes both aggregate and per-condition values.

By default, the evaluator requires a grade for every candidate listed in the judgment file for each condition. With `--allow-incomplete`, it removes unjudged candidates from the returned ranking before scoring and reports judged coverage; this is a condensed-ranking analysis and should be reported as incomplete. `unable_to_judge` has no grade, so it contributes to the candidate denominator and requires this incomplete mode in the current implementation. Do not silently interpret it as grade 0. The controlled synthetic benchmark's author-defined qrels remain its primary evaluation reference.

## Output and reproducibility

Keep each judge's JSONL separate from human annotations. Preserve the raw response and model/prompt/schema metadata in the row. Pin model revisions and vLLM versions before freezing qrels. If a prompt or rubric changes, increment its version and regenerate judgments; cache identity then prevents accidental reuse across versions. Any adjudicated result should retain original judgments and a separate adjudication record.

## Human calibration and audit

Before relying on the scores, humans should judge a stratified sample while blinded to both method and LLM label. Sample across every condition and include likely relevant, likely irrelevant, ambiguous, and randomly selected pairs—not only high-confidence examples. A practical first audit is about 5–10 pairs per condition (125–250 pairs for 25 conditions).

For the current five real conditions, a full catalog is only 195 pairs. If annotation time permits, human-labeling all 195 is an especially strong first reference set; then use it to select and calibrate the LLM judge before scaling to more conditions. If that is too much effort, use the stratified audit above.

Compare LLM and human grades with ordinal agreement (for example, weighted Cohen’s kappa for two raters), grade error/confusion, and agreement on the resulting method rankings. Review disagreements and revise the rubric before freezing the prompt. If the prompt changes, version it and regenerate the affected judgments.

LLM relevance assessment has shown useful agreement with human-based system rankings in some TREC settings, but it is not uniformly reliable: published work reports overrating/query-term effects and cases where LLM assessors favor LLM-based ranking systems. Therefore, use human judgments as the validation reference and do not claim the automated labels are objective ground truth. See the [NIST TREC 2024 RAG study](https://www.nist.gov/publications/large-scale-study-relevance-assessments-large-language-models-initial-look), [SIGIR 2025 comparison of LLM and human relevance assessors](https://eprints.gla.ac.uk/352747/), and [LLM-Assisted Relevance Assessments (LARA)](https://arxiv.org/abs/2411.06877).

## Research references and design takeaways

| Paper | Main takeaway for this project |
| --- | --- |
| [A Large-Scale Study of Relevance Assessments with LLMs: An Initial Look (NIST)](https://www.nist.gov/publications/large-scale-study-relevance-assessments-large-language-models-initial-look) | Automatic LLM judgments can yield system rankings that correlate strongly with human-judgment rankings in a TREC setting, but humans were stricter. Treat this as evidence for scalable evaluation, not proof that individual LLM labels are ground truth. |
| [LLM Relevance Assessors Agree With One Another More Than With Human Assessors (SIGIR 2025)](https://eprints.gla.ac.uk/352747/) | LLM judges can agree with one another while differing from humans, and may favor LLM-based rankers. Compare models, but use independent human annotations as an anchor rather than treating model consensus as truth. |
| [LLM-Assisted Relevance Assessments: When Should We Ask LLMs for Help? (LARA)](https://arxiv.org/abs/2411.06877) | Spend limited human-labeling effort strategically: use human judgments to calibrate LLM predictions, then use calibrated predictions to extend coverage. This motivates human calibration before scaling labels. |
| [Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge](https://arxiv.org/abs/2406.07791) | Judge outputs can be sensitive to presentation position and repetition, with effects varying by model and task. Test stability under prompt/presentation variants; our paraphrases test wording sensitivity specifically, not position bias itself. |
