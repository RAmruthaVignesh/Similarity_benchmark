# Experimental similarity scoring

This mode compares the original SimBench aggregation with a dataset-fitted alternative. It does not interpret condition text, choose feature directions, or assign condition-specific signal weights. Its transformations are fitted only from the training references. The scores are not probabilities of human relevance.

## How it works

For each condition, the reference queries are split into training and held-out groups. A profile is fitted from the training group and evaluated on the held-out group. The candidate pool can overlap across these groups, so this is a query split, not an independent-candidate split.

1. **Score normalization:** for each signal, fit a robust center and scale from its observed training scores. The center is the median; the scale is `1.4826 × MAD`, with IQR and nonzero-deviation fallbacks for discrete or degenerate distributions. Convert the robust z-score `z = (score − median) / scale` to a bounded score with `0.5 + z / (2 × (1 + |z|))`. This is monotonic, maps the training center to `0.5`, and keeps every signal score in `[0, 1]` without a clipping threshold.
2. **Weights:** all selected signals have equal nominal weight. There are no manually assigned or condition-specific weights.
3. **Missing evidence:** the default `reference` policy shrinks missing or low-confidence scores toward that signal’s median bounded training score (normally `0.5`) while keeping all nominal weights in the denominator. It also reports evidence coverage. A `drop` policy is available as a comparison and averages only supported evidence. The optional coverage penalty defaults to zero; final scores are clipped to `[0, 1]`.
4. **No condition rules:** candidate values such as price or verification status are not inspected to reverse or replace pairwise similarity. The original condition files are unchanged.

There is one experimental scoring method: bounded robust-z normalization. It is applied automatically to all conditions and signals without a hand-maintained policy file. The result is a bounded relative-similarity score, not a probability or a calibrated relevance grade.

## Evaluate existing saved scores

From the repository root, in the SimBench environment:

    export TMPDIR=/storage/lcostes/tmp
    export HF_HOME=/storage/lcostes/model-cache/huggingface
    export HF_HUB_CACHE=/storage/lcostes/model-cache/huggingface/hub
    /storage/lcostes/ray-share/venvs/simbench/bin/python scripts/evaluate_experimental_similarity.py

This reads the existing component-score files, so it does not rerun discovery or embeddings. The default output is `results/rel-amazon/similarity/experimental-scoring-bounded-robust-z`, with one directory per condition. Use repeated `--condition-id condition-001` arguments to select conditions. The runner refuses to overwrite profiles already present; choose a new `--output-dir` for a fresh run. Earlier unbounded robust-z results remain in `experimental-scoring-robust-z` and are not loaded by the current notebook.

Each condition directory contains `profile.json`, held-out candidate scores and evidence coverage in `pairs.jsonl`, per-reference ranking comparisons, top-result coverage, and `summary.json`. Open `notebooks/compare_conditioned_entity_rankings.ipynb` to compare bounded robust-z, original SimBench, and available external-baseline rankings on matched held-out references, and inspect entity attributes and evidence side by side.

## Apply a fitted profile to a new ranking

Pass `--experimental-profile /path/to/profile.json` to `scripts/score_entity_similarity.py` with the same dataset, condition, entity table, signals, and similarity measures used to fit the profile. Results retain original and transformed scores, coverage, and per-signal evidence. The experimental profile applies equal signal weights.

## Human comparison

Run `scripts/annotate_similarity_comparisons.py` to compare held-out candidates without displaying which method produced each candidate. It shows full entity attributes, randomizes A/B order reproducibly, spreads cases across conditions, and saves each preference immediately. Disagreement cases compare distinct top candidates; agreement controls compare the shared top candidate with the original ranking’s unique next candidate. Tied tops are skipped.

    /storage/lcostes/ray-share/venvs/simbench/bin/python scripts/annotate_similarity_comparisons.py --max-pairs 30

The default output is `annotations/rel-amazon/entity-comparisons/bounded-robust-z-pairwise.jsonl`. These labels concern entity similarity and are separate from signal-discovery labels.
